function [O_H, O_V, O_H_t, O_V_t] = stINRF3_compute(TCB_matrix, dov, model, framerate, N)
% stINRF3_compute  2D spatiotemporal INRF motion energy computation.
%
%   Inputs
%   TCB_matrix (double, rows x cols x n_frames) input luminance video tensor (values typically 0:255)
%   dov (double) degrees of visual angle covered by the stimulus
%   model (struct) model parameters (from stINRF3_config). Relevant fields: sigmaMu, sigmaW, lambda, p, q, Nm, Nw, sigmag, boundMu, lpa, lpb, lpa2, lpb2, gamma, sf0 or sf0_factor, kernel_type ('gabor'|'sign')
%   framerate (double) temporal sampling rate [Hz]
%   N (double) spatial resolution (number of pixels along the reference axis)
%
% Outputs
%   O_H (double, rows x cols) accumulated horizontal motion energy map
%   O_V (double, rows x cols) accumulated vertical motion energy map
%   O_H_t (double, rows x cols x n_frames (or [] if not requested)) per-frame horizontal motion energy
%   O_V_t (double, rows x cols x n_frames (or [] if not requested)) per-frame vertical motion energy

    dpi         = N / dov;
    sigmaMu_pix = model.sigmaMu * dpi;
    sigmaW_pix  = model.sigmaW  * dpi;

    [rows, cols, n_frames] = size(TCB_matrix);

    % Temporal filter lengths in frames
    Nm_frames = round(model.Nm / 1000 * framerate);
    Nw_frames = round(model.Nw / 1000 * framerate);

    % Build temporal filters
    Tm = makeTm(Nm_frames);
    Tw = makeTw(Nw_frames, model.lpa2, model.lpb2);
    Tb = makeTb(n_frames,  model.lpa,  model.lpb);

    % Gabor spatial frequency [cy/px]
    if isfield(model, 'sf0') && ~isempty(model.sf0)
        sf0_cpx = model.sf0 / dpi;
    elseif isfield(model, 'sf0_factor')
        sf0_cpx = model.sf0_factor / N;
    else
        sf0_cpx = 4 / N;
    end

    gamma = 0.65;
    if isfield(model, 'gamma'), gamma = model.gamma; end
    kernel_size = max(2 * ceil(3 * sigmaW_pix) + 1, 3);

    % Build H/V Gabor (or sign) kernels
    [w_H, w_V] = makeGaborPair(sf0_cpx, sigmaW_pix, gamma, kernel_size, model);

    % Separable factorisation of the Gabor kernels via SVD. An axis-aligned
    [cH, rH] = sepFactors(w_H);
    [cV, rV] = sepFactors(w_V);

    % A-stage: spatial Gaussian smoothing + normalisation to [0,1]
    L_mu = Astage(TCB_matrix, sigmaMu_pix, model.boundMu);

    % Causal temporal filtering (Tm) along the time axis, all pixels at once
    B = applyTm(L_mu, Tm);

    % Precompute the g-smoothing kernel (used inside the frame loop)
    sigmag = model.sigmag;
    if sigmag > 0
        g = fspecial('Gaussian', ceil(3*sigmag), sigmag);
    else
        g = [];
    end

    steps = 25;     % number of luminance quantisation levels

    % Helper grids for vectorised linear-index interpolation (constant)
    MN              = rows * cols;
    [rGrid, cGrid]  = ndgrid(1:rows, 1:cols);
    baseLin         = rGrid + (cGrid - 1) * rows;

    % Output accumulators
    O_H = zeros(rows, cols);
    O_V = zeros(rows, cols);

    save_temporal = (nargout >= 3);
    if save_temporal
        O_H_t = zeros(rows, cols, n_frames);
        O_V_t = zeros(rows, cols, n_frames);
    end

    scale = 1 / (1 + abs(model.lambda));

    % =====================================================================
    %  Main loop over frames
    % =====================================================================
    for tt = 1:n_frames
        Btt = B(:,:,tt);

        % Smooth current frame for level computation (if sigmag > 0)
        if sigmag > 0
            gu = imfilter(Btt, g, 'symmetric');
        else
            gu = Btt;
        end
        lo = min(gu(:));
        hi = max(gu(:));

        degenerate = (hi - lo) < 1e-10;

        % Precompute interpolation indices and fractions for this frame.
        if ~degenerate
            levels     = linspace(lo, hi, steps);
            level_step = levels(2) - levels(1);
            levels3    = reshape(levels, 1, 1, steps);  % [1 x 1 x steps] for broadcasting

            % Bin each pixel into [1, steps-1]; fraction in [0,1]
            idx  = floor((gu - lo) / level_step) + 1;
            idx  = max(1, min(idx, steps - 1));
            frac = (gu - levels(idx)) / level_step;

            % Linear indices into the [rows x cols x steps] response stack
            lin0 = baseLin + (idx - 1) * MN;
            lin1 = lin0 + MN;
        end

        INRF2_H = zeros(rows, cols);
        INRF2_V = zeros(rows, cols);

        % Lag integration (Tw weights)
        if ~degenerate
            for u = 0 : Nw_frames-1
                if (tt - u) >= 1
                    Bprev = B(:,:,tt-u);

                    % Evaluate the nonlinearity for ALL levels at once.
                    % This is shared between H and V channels (same Bprev).
                    S_stack = sigmaNL(levels3 - Bprev, model.p, model.q);

                    % Separable Gabor convolution over the level stack
                    R_L_H = imfilter(imfilter(S_stack, cH, 'symmetric'), rH, 'symmetric');
                    R_L_V = imfilter(imfilter(S_stack, cV, 'symmetric'), rV, 'symmetric');

                    % Interpolate at each pixel's actual luminance value
                    Ru_H = R_L_H(lin0) + (R_L_H(lin1) - R_L_H(lin0)) .* frac;
                    Ru_V = R_L_V(lin0) + (R_L_V(lin1) - R_L_V(lin0)) .* frac;
                    Ru_H(isnan(Ru_H)) = 0;
                    Ru_V(isnan(Ru_V)) = 0;

                    INRF2_H = INRF2_H + Tw(u+1) * Ru_H;
                    INRF2_V = INRF2_V + Tw(u+1) * Ru_V;
                end
            end
        end

        % Combine INRF response with baseline and accumulate
        frame_H = (model.lambda * INRF2_H - Btt) * scale;
        frame_V = (model.lambda * INRF2_V - Btt) * scale;

        O_H = O_H + Tb(tt) * frame_H;
        O_V = O_V + Tb(tt) * frame_V;

        if save_temporal
            O_H_t(:,:,tt) = frame_H;
            O_V_t(:,:,tt) = frame_V;
        end
    end

    if ~save_temporal
        O_H_t = [];
        O_V_t = [];
    end
end


%% ========================================================================
%  Local helper functions
%  ========================================================================

function [wcol, wrow] = sepFactors(w)
% sepFactors  Rank-1 factorisation of a 2D kernel via SVD.
%
%   Inputs
%   w (double, K x K) 2D convolution kernel
%
% Outputs
%   wcol (double, K x 1) column factor (filters along rows, dim 1)
%   wrow (double, 1 x K) row factor (filters along cols, dim 2)
    [U, Sv, V] = svd(w);
    sq   = sqrt(Sv(1,1));
    wcol = U(:,1) * sq;
    wrow = (V(:,1) * sq).';
end


function L_mu = Astage(TCB, sigmaMu_pix, boundMu)
% Astage  A-stage: Gaussian spatial smoothing + luminance normalisation.
%
%   Inputs
%   TCB (double, rows x cols x n_frames) 
%   sigmaMu_pix (double) , Gaussian sigma in pixels
%   boundMu (boundary) condition for imfilter (e.g. 0, 'symmetric')
%
% Outputs
%   L_mu (double, rows x cols x n_frames) , smoothed and /100
    if sigmaMu_pix > 0
        sz = max(3, round(3*sigmaMu_pix));
        if mod(sz,2)==0, sz=sz+1; end
        h    = fspecial('gaussian', sz, sigmaMu_pix);
        L_mu = imfilter(TCB/100, h, boundMu);
    else
        L_mu = TCB / 100;
    end
end


function B = applyTm(L_mu, Tm)
% applyTm  Causal temporal FIR filtering along the time axis.
%
%   Inputs
%   L_mu (double, rows x cols x n_frames) 
%   Tm (double, Nm_frames x 1) , biphasic temporal filter
%
% Outputs
%   B (double, rows x cols x n_frames) , temporally filtered signal
    B = filter(Tm(:), 1, L_mu, [], 3);
end


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


function [w_H, w_V] = makeGaborPair(sf0_cpx, sigmaW_pix, gamma, kernel_size, model)
% makeGaborPair  Build horizontal and vertical spatial filter kernels.
%
%   Inputs
%   sf0_cpx (double) , spatial frequency [cy/px]
%   sigmaW_pix (double) , Gaussian envelope sigma [px]
%   gamma (double) , aspect ratio (Gabor only)
%       kernel_size: integer, kernel side length in pixels.
%   model (struct) , must contain kernel_type ('gabor'|'sign')
%
% Outputs
%   w_H (double, kernel_size x kernel_size) , horizontal kernel
%   w_V (double, kernel_size x kernel_size) , vertical kernel
    ktype = 'gabor';
    if isfield(model, 'kernel_type'), ktype = lower(model.kernel_type); end
    half = floor(kernel_size / 2);
    [x, y] = meshgrid(-half:half, half:-1:-half);
    switch ktype
        case 'gabor'
            sigma_x  = sigmaW_pix;
            sigma_y  = sigmaW_pix / gamma;
            envelope = exp(-0.5 * (x.^2/sigma_x^2 + y.^2/sigma_y^2));
            carrier  = sin(2*pi*x/(1/sf0_cpx) - pi);
            w_H = -envelope .* carrier;
            w_V = imrotate(w_H, 90);
        case 'sign'
            Dx_pix = round(sigmaW_pix);
            window = double(abs(x) <= Dx_pix & abs(y) <= Dx_pix);
            w_H = -sign(x) .* window;
            w_V = -sign(y) .* window;
        otherwise
            error('Unknown kernel_type ''%s''.', ktype);
    end
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
