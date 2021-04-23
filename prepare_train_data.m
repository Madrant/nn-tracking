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
function [samples, results] = prepare_train_data(xr, xn, sample_length, result_length, predict_offset, samples_div, snr_values)
    % Setup default parameters
    if ~exist('predict_offset', 'var')
        predict_offset = 0;
    end

    if ~exist('samples_div', 'var')
        samples_div = 1;
    end

    if ~exist('snr_values', 'var')
        snr_values = [];
    end

    % Check input data
    if length(xr) ~= length(xn)
        fprintf("xr and xn length mismatch:\n");
        fprintf("xr: "); disp(size(xr));
        fprintf("xn: "); disp(size(xn));

        diff = abs(length(xr) - length(xn));
        if length(xr) > length(xn)
            xr = xr(diff + 1:length(xr));
        end

        if length(xn) > length(xr)
            xn = xn(diff + 1:length(xn));
        end
    end

    assert(length(xr) == length(xn));

    assert(sample_length >= 0);
    assert(result_length >= 0);

    assert(sample_length <= length(xr));

    assert(predict_offset >= 0);
    assert(samples_div >= 1);

    % Calculate a number of elements in output array
    if sample_length >= result_length
        train_samples_num = round((length(xr) - predict_offset - (sample_length - 1)) / samples_div);
    else
        train_samples_num = round((length(xr) - predict_offset - (result_length - 1)) / samples_div);
    end

    loops = length(snr_values) + 1;

    samples = zeros(train_samples_num * loops, sample_length);
    results = zeros(train_samples_num * loops, result_length);

    for loop = 1: loops
        % Use external measurements for traning
        if loop > length(snr_values)
            xnt = xn;
        else
        % Generate measurements according to SNR
            snr = snr_values(loop);
            xnt = awgn(xr, snr, 'measured');
        end

        [xrt, xnt] = rescale_data(xr, xnt, 0, 1);

        for n = 1 : train_samples_num
            % Get test sample
            s = n;
            sample = xnt(s: s + sample_length - 1);

            % Get result
            s = n + sample_length - 1;
            result = xrt(s + predict_offset: s + result_length + predict_offset - 1);

            % Save train data set to array
            samples(n + (train_samples_num * (loop - 1)),:) = sample;
            results(n + (train_samples_num * (loop - 1)),:) = result;
        end
    end
end

