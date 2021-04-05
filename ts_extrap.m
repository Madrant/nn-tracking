% Interpolation methods:
%
% linear, nearest, next, previous, pchip, 
% cubic, v5cubic, makima, spline
function outputs = ts_extrap(t, x, Y, method, sample_length)
    result_length = 1;
    samples_num = length(x) - (sample_length + result_length - 1);

    outputs = zeros(result_length, samples_num);

    for n = 1 : length(Y) - (sample_length + result_length - 1)
        test_time = t(n:n + sample_length - 1);
        test_sample = Y(n:n + sample_length - 1);

        extrap_time = t(n + sample_length:n + sample_length + result_length - 1);

        output = interp1(test_time, test_sample, extrap_time, method, 'extrap');

        outputs(:,n) = output;
    end
end
