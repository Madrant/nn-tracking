% Prepare train data set:
%
% - Divide data according to samples_div
% - Noise sample data according to SNR
% - Randomly loss some data
% - Calculate dt
%
% array of struct with fields:
% t, dt, xn, xr
function [data] = prepare_train_data(data, predict_offset, samples_div, obs_period, obs_duration, obs_start, loss_prob, snr)
    % Setup default parameters
    if ~exist('predict_offset', 'var')
        predict_offset = 0;
    end

    if ~exist('samples_div', 'var')
        samples_div = 1;
    end

    if ~exist('obs_period', 'var')
        obs_period = 5;
    end

    if ~exist('obs_duration', 'var')
        obs_duration = 5;
    end

    if ~exist('obs_start', 'var')
        obs_start = 0;
    end

    if ~exist('loss_prob', 'var')
        loss_prob = 0;
    end

    if ~exist('snr', 'var')
        snr = 0;
    end

    % Divide data according to samples div
    data = data(round(1:length(data)/samples_div));

    % Noise reference data
    xn = awgn([data.xr], snr, 'measured');

    % Update xr field
    %c = num2cell(xr);
    %[data(:).xr] = c{:};

    % Create field xn in input data
    %c = num2cell(xn);
    %[data(:).xn] = c{:};

    % Move output data (xr) and shrink input data (t, xn) according to predict offset
    t = [data(1:end - predict_offset).t];
    xn = xn(1:end - predict_offset); % xn is not in data structure yet
    xr = [data(1 + predict_offset:end).xr];

    assert(length(t) == length(xn));
    assert(length(xn) == length(xr));
    
    % Re-create data structure
    data = struct('t', num2cell(t), 'xr', num2cell(xr), 'xn', num2cell(xn));

    % Randomly loss some data
    data = rand_loss(data, loss_prob);

    % Calculate observations dt
    dt = zeros(size([data.t]));

    for n = 2:length(dt)
        dt(n) = data(n).t - data(n - 1).t;
    end

    % Add 'dt' field to data structure
    c = num2cell(dt);
    [data(:).dt] = deal(c{:});
end
