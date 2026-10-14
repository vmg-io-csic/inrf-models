%% demo_inme.m
%
% Demo: INME recovers the direction of a drifting grating, in 2D.
%
% Run with:   matlab -batch "demo_inme"
%
% Runs in about fifteen seconds at the size used here. Scaling up is expensive:
% the 2D engine evaluates the nonlinear sensor at every pixel of every frame,
% so at the 300 x 300 px and 400 frames of the paper the same sweep takes over
% ten minutes.
%
% A sinusoidal grating drifts in three directions, from 0 to 90 degrees. For
% each one the model is run on the full movie and its two motion-opponent
% channels are pooled into a single vector, whose angle is the direction the
% model reports. The error should be well below a degree.
%
% The point of the demo is the shape of the computation, not the numbers: the
% stimulus is a movie, the model returns a response that varies over space and
% time, and the readout collapses it into one vector per stimulus. That is the
% part that IBNN Lite, being purely spatial, does not have.

clear; clc;

%% Model parameters, taken from the grating experiment of the paper
cfg   = stINRF3_config('gratings');
model = cfg.model;

%% Stimulus
N              = 150;                 % canvas side [px]
dov            = 300 * (1.5/128);     % field of view [deg]
framerate      = 200;                 % [fps]
duration_ms    = 500;
f_spatial_cpd  = 2.0;                 % spatial frequency [cycles/deg]
speed_Hz       = 8;                   % temporal frequency [Hz]
contrast       = 1;
meanL          = 100;
directions_deg = 0:45:90;

%% Readout: Gaussian pooling over a region centred on the stimulus
half_region = 50;
sigma_c     = 2*half_region / sqrt(12);

%% Derived
n_frames = round(duration_ms / 1000 * framerate);
f_cpx    = f_spatial_cpd / (N / dov);     % [cycles/px]
speed    = speed_Hz / framerate;          % [cycles/frame]

fprintf('\n==== INME: direction of a drifting grating ====\n');
fprintf('  %d x %d px, %.2f deg, %d fps, %d frames\n', N, N, dov, framerate, n_frames);
fprintf('  grating at %.1f cyc/deg drifting at %.1f Hz\n\n', f_spatial_cpd, speed_Hz);
fprintf('  %-12s %-12s %s\n', 'stimulus', 'predicted', 'error');
fprintf('  %s\n', repmat('-', 1, 36));

errors = zeros(size(directions_deg));
for k = 1:numel(directions_deg)
    % One movie: (N x N x n_frames) of luminance values.
    TCB = make_grating(N, n_frames, f_cpx, speed, directions_deg(k), ...
                       0, contrast, meanL);

    % The model. O_H and O_V are the horizontal and vertical motion-opponent
    % channels, already integrated over time by the Tb window.
    [O_H, O_V] = stINRF3_compute(TCB, dov, model, framerate, N);

    % The readout: one direction per stimulus.
    predicted = stINRF3_readout_gaussian(O_H, O_V, sigma_c, ...
                                         'SupportHalf', half_region);

    err = abs(mod(predicted - directions_deg(k) + 180, 360) - 180);
    errors(k) = err;
    fprintf('  %8.1f deg %8.1f deg %7.2f deg\n', directions_deg(k), predicted, err);
end

fprintf('  %s\n', repmat('-', 1, 36));
fprintf('  mean absolute error: %.2f deg\n\n', mean(errors));
