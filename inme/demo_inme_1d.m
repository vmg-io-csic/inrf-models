%% demo_inme_1d.m
%
% Demo: the sign of the INME response gives the direction of motion.
%
% Run with:   matlab -batch "demo_inme_1d"
%
% This is the quick demo. The stimulus varies along one spatial axis only, so
% the one-dimensional engine applies and the whole thing runs in seconds. For
% the two-dimensional model, which recovers an arbitrary direction in the
% plane, see demo_inme.m.
%
% A sinusoidal grating drifts rightward and then leftward. The model returns a
% signed response: positive for rightward motion, negative for leftward. The
% two should be close to equal and opposite, because the only difference
% between the stimuli is the sign of the velocity.

clear; clc;

%% Model parameters
cfg   = stINRF3_config('Cont');
model = cfg.model;

%% Stimulus
N            = 400;        % spatial samples [px]
dov          = 4.0;        % field of view [deg]
framerate    = 120;        % [fps]
duration_ms  = 500;
f_cpd        = 1.0;        % spatial frequency [cycles/deg]
speed_dps    = 8.0;        % drift speed [deg/s]
contrast     = 0.8;
meanL        = 100;

%% Readout: Gaussian pooling over the 200 central sensors
sigma_c   = 57.74;
n_frames  = round(duration_ms / 1000 * framerate);
j0        = round(N/2);
j_sensors = (j0-99):(j0+100);

x_deg = (0:N-1) * dov / N - dov/2;
t_sec = (0:n_frames-1) / framerate;

fprintf('\n==== INME: direction of a drifting grating (1D) ====\n');
fprintf('  %d samples over %.1f deg, %d fps, %d frames\n', N, dov, framerate, n_frames);
fprintf('  grating at %.1f cyc/deg drifting at %.1f deg/s\n\n', f_cpd, speed_dps);
fprintf('  %-12s %s\n', 'direction', 'response');
fprintf('  %s\n', repmat('-', 1, 30));

responses = zeros(1, 2);
labels    = {'rightward', 'leftward'};
for k = 1:2
    v = speed_dps * (3 - 2*k);          % +speed, then -speed

    % Build the movie: rows are frames, columns are spatial positions.
    mI = zeros(n_frames, N);
    for t = 1:n_frames
        mI(t, :) = meanL * (1 + contrast * ...
                   sin(2*pi * f_cpd * (x_deg - v * t_sec(t))));
    end

    O_sensors    = stINRF3_compute_1D(mI, dov, model, framerate, N, j_sensors);
    responses(k) = stINRF3_readout_gaussian_1D(O_sensors, j_sensors, sigma_c);

    fprintf('  %-12s %+.4f\n', labels{k}, responses(k));
end

fprintf('  %s\n', repmat('-', 1, 30));
fprintf('  sum of the two: %+.4f  (should be near zero)\n\n', sum(responses));
