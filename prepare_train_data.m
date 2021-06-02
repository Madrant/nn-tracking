% Prepare train data set
%
% xr - real data
% xn - noised measurements
%
% Test function:
%
% xr = [1 2 3 4 5 6 7 8 9 10]
% xn = xr;
% [a,b] = prepare_train_data(xr, xr, 1, 1, 0, 1, []); hold on; plot(a, '-x'); plot(b, '-d'); hold off; disp(size(a)); disp(a.'); disp(size(b)); disp(b.');
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

    % Move output data (xr) and shrink input data (t, xn) according to predict offset
    t = [data(1:end - predict_offset).t];
    xn = [data(1:end - predict_offset).xn];
    xr = [data(1 + predict_offset:end).xr];

    % Re-create data structure
    data = struct('t', num2cell(t), 'xr', num2cell(xr), 'xn', num2cell(xn));

    % Randomly loss some data
    data = rand_loss(data, loss_prob);

    % Calculate observations dt
    dt = zeros(size([data.t]));

    for n = 2:length(dt)
        dt(n) = data(n).t - data(n - 1).t;
    end

    c = num2cell(dt);
    [data(:).dt] = deal(c{:});
end
