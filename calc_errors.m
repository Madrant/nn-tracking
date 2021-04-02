function [error, abs_error, mse_array, rmse_array] = calc_errors(x, X)
    error = (x - X);
    abs_error = abs(error);

    mse_array = zeros(length(error));
    rmse_array = zeros(length(error));

    for n=1:length(error)
        mse = mean(error(1:n).^2);
        mse_array(n) = mse;
        rmse_array(n) = sqrt(mse);
    end
end


