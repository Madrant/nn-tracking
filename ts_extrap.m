% Interpolation methods:
%
% linear, nearest, next, previous, pchip, 
% cubic, v5cubic, makima, spline
function outputs = ts_extrap(t, x, method, sample_length, predict_offset)
    if ~exist('method', 'var')
        method = 'spline';
    end

    if ~exist('sample_length', 'var')
        sample_length = 5;
    end

    if ~exist('predict_offset', 'var')
        predict_offset = 1;
    end

    result_length = 1;
    samples_num = length(x) - (sample_length + predict_offset - 1);

    outputs = zeros(result_length, samples_num);

    for n = 1 : samples_num
        test_time = t(n:n + sample_length - 1);
        test_sample = x(n:n + sample_length - 1);

        extrap_time = t(n + sample_length - 1 + predict_offset);

        output = interp1(test_time, test_sample, extrap_time, method, 'extrap');

        outputs(:,n) = output;
    end
end
