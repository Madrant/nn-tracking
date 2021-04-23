function [error, abs_error, mse_array, rmse_array, max_error, mean_error, mse, rmse] = calc_errors(x, X)
    error = (x - X);
    abs_error = abs(error);

    mse_array = zeros(1, length(error));
    rmse_array = zeros(1, length(error));

    for n = 1:length(error)
        mse = mean(error(1:n).^2);
        mse_array(1, n) = mse;
        rmse_array(1, n) = sqrt(mse);
    end

    max_error = max(abs_error);
    mean_error = mean(abs_error);
    mse = mean(error.^2);
    rmse = sqrt(mse);
end
