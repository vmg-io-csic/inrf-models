function cfg = stINRF3_config(experiment)
% STINRF3_CONFIG  Central configuration for all stINRF3 experiments.
%
% Inputs
%   experiment (char) one of 'gratings', 'plaidI', 'plaidII', 'DH',
%              'Cont', 'timereversal', 'flash', 'rotating_snakes'
%
% Outputs
%   cfg (struct) with three sub-structs:
%       cfg.model   — model parameters (spatial units in dva, temporal in ms)
%       cfg.stim    — stimulus parameters for the requested experiment
%       cfg.readout — readout/ROI parameters

% -------------------------------------------------------------------------
%  SHARED MODEL PARAMETERS  (Tabla 1 — gratings/plaids/snakes set)
% -------------------------------------------------------------------------
model = struct( ...
    'sf0',        2.0,   ... % [cpd]  preferred spatial frequency
    'sigmaW',     0.15,  ... % [dva]  Gabor envelope sigma
    'sigmaMu',    0.05,  ... % [dva]  pre-smoothing sigma (A stage)
    'lambda',     30,    ... % []     nonlinear gain
    'p',          0.20,  ... % []     sigmaNL threshold shape
    'q',          0.10,  ... % []     sigmaNL saturation
    'Nm',         125,   ... % [ms]   causal Tm duration
    'Nw',         15,    ... % [ms]   Tw integration window
    'sigmag',     0,     ... % []     g-smoothing (0 = off)
    'boundMu',    0,     ... % []     A-stage boundary (0 = zero-pad)
    'lpa',        9,     ... % []     Tb beta exponent a
    'lpb',        1,     ... % []     Tb beta exponent b
    'lpa2',       9,     ... % []     Tw beta exponent a
    'lpb2',       1,     ... % []     Tw beta exponent b
    'gamma',      0.65,  ... % []     Gabor aspect ratio
    'kernel_type','gabor' ); % []     spatial kernel type

% -------------------------------------------------------------------------
%  PER-EXPERIMENT CONFIGURATION
% -------------------------------------------------------------------------
switch lower(experiment)

    % =====================================================================
    case 'gratings'
    % Single sinusoidal gratings moving in different directions.
    % =====================================================================
        stim = struct( ...
            'N',             300,    ... % image size [px]  (square)
            'dov',           300*(1.5/128), ... % field of view [dva]
            'framerate',     200,    ... % [fps]
            'duration_ms',   2000,   ... % stimulus duration [ms]
            'f_spatial',     2.0,    ... % spatial frequency [cpd]
            'speed_cpd_fps', 8,      ... % temporal frequency = speed [cy/frame at framerate]
            'directions_deg', 0:15:90, ... % grating directions to test [deg]
            'n_phase_trials', 5,     ... % random-phase repetitions
            'rng_seed',       42     );  % random seed for reproducibility

        readout = struct( ...
            'half_region',   100,    ... % ROI half-size [px] → 200×200 ROI
            'sensor_mode',   'multisensor' ); % 'single' | 'multisensor'

    % =====================================================================
    case 'plaidi'
    % Type-I plaids: two gratings at ±45° from rigid direction.
    % =====================================================================
        stim = struct( ...
            'N',             300,    ...
            'dov',           300*(1.5/128), ...
            'framerate',     200,    ...
            'duration_ms',   2000,   ...
            'f_spatial',     2.0,    ... % [cpd] spatial frequency of each component
            'speed_cpd_fps', 8,      ... % temporal frequency
            'angle_offset',  45,     ... % each component is ±angle_offset from rigid dir
            'rigid_directions_deg', 0:15:90, ... % rigid motion directions to test
            'n_phase_trials', 5,     ...
            'rng_seed',       42     );

        readout = struct( ...
            'half_region',   100,    ...
            'sensor_mode',   'multisensor' );

    % =====================================================================
    case 'plaidii'
    % Type-II plaids: VA ≠ IOC. Human perception follows IOC.
    % =====================================================================
        stim = struct( ...
            'N',             300,    ...
            'dov',           300*(1.5/128), ...
            'framerate',     200,    ...
            'duration_ms',   2000,   ...
            'f_spatial',     2.0,    ... % [cpd] base spatial frequency
            ... % configs: [speed1, speed2, theta1_deg, theta2_deg, ioc_dir_deg]
            'configs', [ ...
                1,   2,  -45,  -25,   27.12;  ... % Config 1
                1,   2,   50,   35,  -25.95;  ... % Config 2
                1,   2,  -50,  -30,   22.12], ... % Config 3
            'n_phase_trials', 5,     ...
            'rng_seed',       42     );
        % tf_scale converts speed [cy/frame] to match f_spatial:
        % tf_scale = f_spatial / 0.21333  (derived from original grating speed)

        readout = struct( ...
            'half_region',   100,    ...
            'sensor_mode',   'multisensor' );

    % =====================================================================
    case 'dh'
    % Derrington & Henning (1987) / Interaction among scales.
    % =====================================================================
        model.sf0    = 1.50;  % override: DH uses different sf
        model.sigmaW = 0.30;  % override: larger receptive field
        model.p      = 5.0;   % override: DH parameter set
        model.q      = 0.10;
        model.Nm     = 63;    % [ms]
        model.Nw     = 40;    % [ms]
        model.lpa    = 9;
        model.lpa2   = 1;
        model.boundMu = 'symmetric';

        stim = struct( ...
            'N',          400,    ... % [px]  wider image for better coverage
            'N_rows',     50,     ... % [px]  image height (stimulus is 1D → broadcast)
            'dov',        4.0,    ... % [dva] field of view
            'framerate',  1000,   ... % [fps] 1 ms resolution
            'duration_ms', 500,   ... % [ms]
            'speed_dps',  12,     ... % [°/s] grating speed
            ... % Conditions: [sfm, sfs]  (moving SF, static SF)
            'conditions', struct( ...
                'name',  {{'S1','S3','CS','CSr','CM'}}, ...
                'label', {{'simple 1cpd','simple 3cpd', ...
                           'complex: 1mov+3stat', ...
                           'complex: 3mov+1stat', ...
                           'complex: 1mov+3mov'}}, ...
                'sfm',   {[1, 3, 1, 3, 3]}, ...   % moving component SF [cpd]
                'sfs',   {[1, 1, 3, 1, 1]}), ...  % static component SF [cpd]
            'sigma_t_ms', [10, 25, 37.5, 100], ... % Gaussian temporal envelope σ [ms]
            'n_phase_trials', 50,  ... % random-phase repetitions (Npsi)
            'rng_seed',       0    );  % 0 = unseeded (random each run)

        readout = struct( ...
            'half_region',  100,   ... % ROI half-size in x [px]
            'sensor_mode',  'multisensor' );

    % =====================================================================
    case 'cont'
    % Shared 1D parameter set for the contrast, masking, missing-fundamental,
    % opponency, reverse-phi and second-order experiments.
    % =====================================================================
    % Contrast saturation and opponency experiments.
    % Moving sinusoide with different contrasts
    % =====================================================================
        stim = struct( ...
            'N',          400,    ... % [px]  wider image for better coverage
            'N_rows',     50,     ... % [px]  image height (stimulus is 1D → broadcast)
            'dov',        4.0,    ... % [dva] field of view
            'framerate',  1000,   ... % [fps] 1 ms resolution
            'duration_ms', 500,   ... % [ms]
            'speed_dps',  12,     ... % [°/s] grating speed
            'conditions', struct( ...
                'name',  {{'S1'}}, ...
                'label', {{'simple cpd'}}), ...  % static component SF [cpd]
            'sigma_t_ms', [100], ... % Gaussian temporal envelope σ [ms]
            'n_phase_trials', 1,  ... % random-phase repetitions (Npsi)
            'rng_seed',       0    );  % 0 = unseeded (random each run)
        readout = struct( ...
            'half_region',  100,   ... % ROI half-size in x [px]
            'sensor_mode',  'multisensor' );
    % =====================================================================
    % =====================================================================
    case 'flash'
    % Static flash: disc appears and disappears.
    % =====================================================================
        model.sf0    = [];          % use sf0_factor instead
        model = rmfield(model,'sf0');
        model.sf0_factor = 3;       % sf0 = 3/N cy/px
        model.sigmaW = 0.15;
        model.sigmaMu = 0.05;
        model.p      = 0.05;
        model.q      = 0.50;
        model.Nm     = 40;          % [ms]  short Tm → crisp transient
        model.Nw     = 100;         % [ms]
        model.lpa    = 9;
        model.lpa2   = 1;
        model.lpb    = 1;
        model.lpb2   = 1;
        model.boundMu = 0;

        stim = struct( ...
            'N',               128,   ... % [px]
            'dov',             1.5,   ... % [dva]
            'framerate',       1000,  ... % [fps]  1 ms resolution
            'duration_ms',     500,   ... % total stimulus duration [ms]
            'flash_onset_ms',  150,   ... % flash onset [ms]
            'flash_dur_ms',    100,   ... % flash duration [ms]
            'mean_luminance',  100,   ... % background luminance [a.u.]
            'disc_amplitude',  50,    ... % flash increment [a.u.]
            'disc_radius_deg', 0.3,   ... % disc radius [dva]
            'envelope_type',   'square' ); % 'square' | 'gaussian'
        % For gaussian envelope, sigma = flash_dur_ms / 4.

        readout = struct( ...
            'half_region',   25,    ... % ROI half-size [px] → 50×50 centre
            'ring_width_px', 5,     ... % width of ring mask around disc edge
            'sensor_mode',   'ring' );  % 'ring' extracts mean on disc boundary

    % =====================================================================
    case 'rotating_snakes'
    % Rotating Snakes illusion (Kitaoka & Ashida, 2003).
    % =====================================================================
        model.sf0    = [];
        model = rmfield(model,'sf0');
        model.sf0_factor = 3;       % sf0 = 3/N cy/px  (same as grating set)
        model.sigmaW = 0.15;
        model.sigmaMu = 0.05;
        model.p      = 0.20;
        model.q      = 0.10;
        model.Nm     = 125;
        model.Nw     = 15;
        model.lpa    = 9;  model.lpb  = 1;
        model.lpa2   = 9;  model.lpb2 = 1;
        model.boundMu = 0;

        stim = struct( ...
            'N',              128,   ... % [px]
            'dov',            1.5,   ... % [dva]
            'framerate',      1000,  ... % [fps]
            'onset_ms',       150,   ... % stimulus onset [ms]
            'duration_ms',    300,   ... % stimulus duration [ms]
            'post_offset_ms', 200,   ... % frames after offset
            ... % Sawtooth luminance profile: [dark_a, light_a, dark_b, light_b]
            ... % Illusion (asymmetric): [0, 5, 100, 50]
            ... % Control  (symmetric):  [0, 100, 0, 100]
            'L_illusion',  [0, 5, 100, 50],   ...
            'L_symmetric', [0, 100, 0, 100],  ...
            'mean_luminance', 50,    ...
            'orientations_deg', [0, 90], ... % pattern orientations to test
            'speed_dps',   [],       ... % real-motion speed [°/s]; [] = auto from sf0/Nm
            'conditions',  {{'illusion','symmetric','real_motion','baseline'}} );

        readout = struct( ...
            'half_region',  25,    ... % ROI half-size [px] → 50×50
            'sensor_mode',  'multisensor' );

    % =====================================================================
    case 'timereversal'
    % Time-reversal symmetry-breaking stimuli.
    % =====================================================================
    % Time-reversal symmetry-breaking stimuli. Two periodic profiles
    % (LightEdges: single sharp transition per period; UnevenSquareWave:
    % bars in ratio 5:5:25:25) drifting at their paper-defined
    % velocity/period. Time-reversing the frame order (NOT flipping
    % drift direction) breaks symmetry in a biological motion detector;
    % reference fly ratios |V|/|I| are 5.72 (LightEdges) and 12.74
    % (UnevenSquareWave). Uses the shared model set (Tabla 1).
    % =====================================================================
        stim = struct( ...
            'N',             300,    ... % [px] spatial samples
            'dov',           300*(1.5/128), ... % [dva] field of view
            'framerate',     1000,   ... % [fps] canonical stINRF3 rate
            'meanL',         50,     ... % mean luminance [a.u.]
            'contrast',      1,      ... % Michelson-like contrast
            'phase0_rad',    0,      ... % deterministic spatial phase
            ... % Paper velocities & periods (do NOT modify)
            'v_LightEdges',       9.38,  ... % [deg/s]
            'period_LightEdges',  1.17,  ... % [deg]
            'v_UnevenSquareWave', 1.17,  ... % [deg/s]
            'period_UnevenSquareWave', 2.34, ... % [deg]
            'dur_ms_list',   [200, 500, 1000], ... % durations to sweep [ms]
            'ratio_fly_LightEdges',       5.72,  ... % reference fly |V|/|I|
            'ratio_fly_UnevenSquareWave', 12.74  ); % reference fly |V|/|I|
        readout = struct( ...
            'half_region',   100,    ... % ROI half-size [px] → 200 sensors
            'sensor_mode',   'multisensor' );
    % =====================================================================
    otherwise
        error('stINRF3_config: unknown experiment ''%s''.\n  Valid options: gratings | plaidI | plaidII | DH | Cont | timereversal | flash | rotating_snakes', experiment);
end

% -------------------------------------------------------------------------
%  OUTPUT
% -------------------------------------------------------------------------
cfg.model   = model;
cfg.stim    = stim;
cfg.readout = readout;

end
