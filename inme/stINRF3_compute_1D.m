function [O_sensors, O_frames] = stINRF3_compute_1D(stim_1D, dov, model, framerate, N, j_sensors)
% stINRF3_compute_1D  1D spatiotemporal INRF motion energy computation.
%
%   Inputs
%   stim_1D (double, T x N) 1D stimulus matrix: rows are time frames, columns are spatial positions. Values typically in [0, 255]
%   dov (double) degrees of visual angle spanned by the stimulus
%   model (struct) model parameters (from stINRF3_config). Relevant fields: sigmaMu, sigmaW, lambda, p, q, Nm, Nw, lpa, lpb, lpa2, lpb2, gamma, sf0 or sf0_factor
%   framerate (double) temporal sampling rate [Hz]
%   N (double) number of spatial samples (columns in stim_1D)
%   j_sensors (integer, 1 x Nsensors (optional)) column indices at which to read out the motion energy Default: 200 sensors centred at round(N/2)
%
% Outputs
%   O_sensors (double, 1 x Nsensors) accumulated motion energy at each sensor position
%   O_frames (double, T x Nsensors) per-frame response before the Tb temporal integration, so that O_sensors = sum_t Tb(t) * O_frames(t,:). Use it to plot response time courses

    [T, ~] = size(stim_1D);
    dpi    = N / dov;

    % Default sensor positions (200 centred sensors)
    if nargin < 6 || isempty(j_sensors)
        j0        = round(N/2);
        j_sensors = (j0-100) : (j0+99);
    end
    j_sensors = j_sensors(:).';
    Nsensors  = numel(j_sensors);

    % Temporal filter lengths in frames
    Nm_frames = round(model.Nm / 1000 * framerate);
    Nw_frames = round(model.Nw / 1000 * framerate);

    % Build temporal filters
    Tm = makeTm(Nm_frames);
    Tw = makeTw(Nw_frames, model.lpa2, model.lpb2);
    Tb = makeTb(T,         model.lpa,  model.lpb);

    % 1D Gabor kernel
    sigmaW_px = model.sigmaW * dpi;
    if isfield(model, 'sf0') && ~isempty(model.sf0)
        f_cpx = model.sf0 / dpi;
    elseif isfield(model, 'sf0_factor')
        f_cpx = model.sf0_factor / N;
    else
        f_cpx = 4 / N;
    end
    half_k = fix(ceil(3 * sigmaW_px));
    m      = -half_k : half_k;
    wv     = -exp(-0.5 * m.^2 / sigmaW_px^2) .* sin(2*pi * f_cpx * m - pi);

    % The 2D kernel is separable, w(x,y) = wv(x) * exp(-y^2/(2*sigma_y^2))
    % with sigma_y = sigmaW_px/gamma. For a stimulus that does not vary
    % along y the correlation along y collapses to the constant below, so
    % including it makes this engine the central row of stINRF3_compute.
    gamma_ar = 0.65;
    if isfield(model, 'gamma'), gamma_ar = model.gamma; end
    sigma_y  = sigmaW_px / gamma_ar;
    my       = -floor(max(2*ceil(3*sigmaW_px) + 1, 3) / 2) : ...
                floor(max(2*ceil(3*sigmaW_px) + 1, 3) / 2);
    wv       = wv * sum(exp(-0.5 * my.^2 / sigma_y^2));

    % A-stage: 1D Gaussian smoothing + /100, applied to the full [T x N]
    % matrix at once (a 1xK kernel filters each row independently)
    sigmaMu_px = model.sigmaMu * dpi;
    if sigmaMu_px > 0
        sz_mu = max(3, round(3*sigmaMu_px));
        if mod(sz_mu,2)==0, sz_mu=sz_mu+1; end
        h_mu = fspecial('gaussian', [1, sz_mu], sigmaMu_px);
        L_mu = imfilter(stim_1D/100, h_mu, 'symmetric');
    else
        L_mu = stim_1D / 100;
    end

    % Causal temporal filtering: filter() operates down dim 1 by default,
    % so this processes all spatial columns simultaneously
    B = filter(Tm(:), 1, L_mu);

    % Nonlinear temporal integration
    nLevels   = 25;
    O_sensors = zeros(1, Nsensors);
    O_frames  = zeros(T, Nsensors);
    scale     = 1 / (1 + abs(model.lambda));

    for tt = 1:T
        B_sensors = B(tt, j_sensors);
        lo = min(B_sensors);
        hi = max(B_sensors);
        if (hi - lo) < 1e-12, continue; end

        levels     = linspace(lo, hi, nLevels);
        level_step = levels(2) - levels(1);

        % Interpolation indices for every sensor
        vals = B(tt, j_sensors);
        li   = floor((vals - lo) / level_step) + 1;
        li   = max(1, min(li, nLevels-1));
        frc  = (vals - levels(li)) / level_step;

        % Linear indices into the [nLevels x N] response array
        gobase = li + (j_sensors - 1) * nLevels;
        gonext = gobase + 1;

        NL_sensors = zeros(1, Nsensors);

        umax = min(Nw_frames-1, tt-1);
        for u = 0 : umax
            row_prev = B(tt-u, :);

            % Evaluate sigmaNL for all levels at once via broadcasting:
            % (1 x N) minus (nLevels x 1) gives [nLevels x N]
            S = sigmaNL(row_prev - levels.', model.p, model.q);

            % Gabor convolution over the level stack (1xK kernel filters
            % each row of the [nLevels x N] matrix independently)
            R_L = imfilter(S, wv, 'symmetric');

            % Interpolate at each sensor position
            NL_sensors = NL_sensors + ...
                Tw(u+1) * (R_L(gobase) + frc .* (R_L(gonext) - R_L(gobase)));
        end

        % Combine INRF response with baseline and accumulate over time
        frame_val = (model.lambda * NL_sensors - B(tt, j_sensors)) * scale;
        O_frames(tt, :) = frame_val;
        O_sensors = O_sensors + Tb(tt) * frame_val;
    end
end


%% ========================================================================
%  Local helper functions
%  ========================================================================

function s = sigmaNL(x, p, q)
% sigmaNL  Pointwise saturating nonlinearity (INRF core).
%
%   Inputs
%   x (double) array (any size)
%   p (double) , steepness parameter
%   q (double) , saturation parameter
%
% Outputs
%   s (double) array (same size as x), normalised nonlinear response
    sMax = (1 - exp(-1/p^2)) * 1^p / (1^p + q);
    s    = (1 - exp(-x.^2/p^2)) .* abs(x).^p ./ (abs(x).^p + q);
    s    = s / sMax;
    s(isnan(s)) = 0;
end


function Tm = makeTm(Nm_frames)
% makeTm  Causal biphasic temporal filter.
%
%   Inputs
%   Nm_frames (integer) , filter length in frames
%
% Outputs
%   Tm (double, 1 x Nm_frames) , normalised filter coefficients
    t  = linspace(0, 20, Nm_frames);
    k  = 1;  n = 5;
    Tm = (k*t).^n .* exp(-k*t) .* (1/factorial(n) - (k*t).^2/factorial(n+2));
    Tm = Tm / (sum(abs(Tm)) + eps);
end


function Tw = makeTw(Nw_frames, lpa2, lpb2)
% makeTw  Temporal lag weights (beta-shaped window).
%
%   Inputs
%   Nw_frames (integer) , number of lags
%       lpa2, lpb2: double scalars, shape parameters.
%
% Outputs
%   Tw (double, 1 x Nw_frames) , normalised lag weights
    z  = 0 : Nw_frames-1;
    Tw = (z/Nw_frames).^lpa2 .* (1 - z/Nw_frames).^lpb2;
    Tw = Tw / (sum(Tw) + eps);
end


function Tb = makeTb(n_frames, lpa, lpb)
% makeTb  Temporal pooling window (beta-shaped).
%
%   Inputs
%   n_frames (integer) , total number of frames
%       lpa, lpb  : double scalars, shape parameters.
%
% Outputs
%   Tb (double, 1 x n_frames) , normalised pooling weights
    t  = 0 : n_frames-1;
    Tb = (t/n_frames).^lpa .* (1 - t/n_frames).^lpb;
    Tb = Tb / (sum(Tb) + eps);
end
