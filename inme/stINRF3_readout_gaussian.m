function [direction_deg, magnitude, info] = stINRF3_readout_gaussian(O_H, O_V, sigma_c, varargin)
% STINRF3_READOUT_GAUSSIAN  Spatial readout with the Gaussian filter c_s.
%
% Inputs
%   O_H (double, rows x cols)  horizontal channel (Tb-integrated)
%   O_V (double, rows x cols)  vertical channel (Tb-integrated)
%   sigma_c (double)           s_min [px].  Scalar -> global Gaussian.
%
% Outputs
%   direction_deg (double)  motion direction [deg], 0 = right, 90 = up
%   magnitude (double)      sqrt(mean_OH^2 + mean_OV^2)
%   info (struct)           .mean_OH, .mean_OV, .W (weight map),
%                           .n_eff (effective number of pixels),
%                           .sigma_map (s_c(x,y))

    p = inputParser;
    p.addParameter('SupportHalf', 100);
    p.addParameter('RowHalf',     []);
    p.addParameter('Centre',      []);
    p.addParameter('SigmaMax',    []);
    p.addParameter('RadiusT',     0);
    p.addParameter('Sharpness',   1);
    p.parse(varargin{:});
    o = p.Results;

    [rows, cols] = size(O_H);
    if isempty(o.Centre)
        cx = round(cols/2) + 0.5;   % same convention as the square ROI
        cy = round(rows/2) + 0.5;   % (centre of the [c-h+1 : c+h] block)
    else
        cx = o.Centre(1);  cy = o.Centre(2);
    end
    if isempty(o.RowHalf),  o.RowHalf  = o.SupportHalf;  end
    if isempty(o.SigmaMax), o.SigmaMax = sigma_c;        end

    [XX, YY] = meshgrid(1:cols, 1:rows);
    dx = XX - cx;
    dy = YY - cy;
    r2 = dx.^2 + dy.^2;

    % --- spatially variable width s_c(x,y) -------------------------------
    if o.SigmaMax == sigma_c
        s_c = sigma_c * ones(rows, cols);
    else
        r_c = sqrt(r2);
        s_c = sigma_c + (o.SigmaMax - sigma_c) ./ ...
              (1 + exp((r_c - o.RadiusT) / o.Sharpness));
    end

    % --- Gaussian weights -------------------------------------------------
    W = 1 ./ (2*pi*s_c.^2) .* exp(-r2 ./ (2*s_c.^2));

    % --- support (truncation) --------------------------------------------
    if isfinite(o.SupportHalf)
        W(abs(dx) > o.SupportHalf | abs(dy) > o.RowHalf) = 0;
    end

    W = W / sum(W(:));                       % kappa normalisation

    mean_OH = sum(W(:) .* O_H(:));
    mean_OV = sum(W(:) .* O_V(:));

    direction_deg = atan2d(mean_OV, mean_OH);
    magnitude     = sqrt(mean_OH^2 + mean_OV^2);

    info.mean_OH   = mean_OH;
    info.mean_OV   = mean_OV;
    info.W         = W;
    info.n_eff     = 1 / sum(W(:).^2);       % effective number of pixels
    info.sigma_map = s_c;
end
