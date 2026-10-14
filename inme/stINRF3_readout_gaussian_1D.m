function [val, info] = stINRF3_readout_gaussian_1D(O_sensors, j_sensors, sigma_c, varargin)
% STINRF3_READOUT_GAUSSIAN_1D  Central row of the 2D Gaussian readout.
%
% Inputs
%   O_sensors (double, 1 x Nsensors)  output of stINRF3_compute_1D
%   j_sensors (double, 1 x Nsensors)  the sensor column indices that were
%                                     passed to stINRF3_compute_1D
%   sigma_c (double)                  s_c in pixels
%
% Outputs
%   val (double)   weighted mean response (sign = reported direction)
%   info (struct)  .w (weights), .n_eff, .flat_val (unweighted mean)

    p = inputParser;
    p.addParameter('Centre', []);
    p.addParameter('SupportHalf', Inf);
    p.parse(varargin{:});
    o = p.Results;

    O_sensors = O_sensors(:).';
    j_sensors = j_sensors(:).';
    if isempty(o.Centre)
        o.Centre = mean(j_sensors);
    end

    x = j_sensors - o.Centre;

    % c_s(x, 0): central row of the 2D isotropic Gaussian sensor
    w = 1/(2*pi*sigma_c^2) * exp(-x.^2 / (2*sigma_c^2));

    w(abs(x) > o.SupportHalf) = 0;
    w = w / sum(w);                       % kappa normalisation

    val = sum(w .* O_sensors);

    info.w        = w;
    info.n_eff    = 1 / sum(w.^2);
    info.flat_val = mean(O_sensors);
end
