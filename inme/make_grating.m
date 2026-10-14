function TCB = make_grating(N, n_frames, f_cpx, speed, direction_deg, ...
                             phase0_rad, contrast, meanL)
% MAKE_GRATING  Generate a drifting sinusoidal grating stimulus.
%
% Inputs
%   N (int) image size in pixels (square: N x N)
%   n_frames (int) number of temporal frames
%   f_cpx (double) spatial frequency [cycles/pixel]
%   speed (double) temporal frequency [cycles/frame]
%                            Computed in run scripts as speed_Hz / framerate.
%   direction_deg (double) motion direction [degrees]
%   phase0_rad (double) initial spatial phase [radians]
%   contrast (double) michelson contrast, typically 1
%   meanL (double) mean luminance [a.u.], typically 100
%
% Outputs
%   TCB (single, N x N x n_frames) stimulus tensor
%        Luminance values in [0, 2*meanL] (for contrast=1).

    theta  = deg2rad(direction_deg);
    coords = linspace(-N/2, N/2, N);
    [X, Y] = meshgrid(coords, coords);
    X_th   = -X*cos(theta) + Y*sin(theta);

    TCB = zeros(N, N, n_frames, 'single');
    for t = 1:n_frames
        phase_t    = phase0_rad + 2*pi * speed * (t-1);
        TCB(:,:,t) = single(meanL * (1 + contrast * sin(2*pi * f_cpx * X_th + phase_t)));
    end
end
