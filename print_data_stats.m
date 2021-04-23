function print_data_stats(xr, xn)
    fprintf("Measurements Mean error: %f\n", mean(abs(xr - xn)));
    fprintf("Measurements Max error:  %f\n", max(abs(xr - xn)));
    fprintf("Measurements MSE:        %f\n", mean(xr - xn).^2);
    fprintf("Measurements RMSE:       %f\n", sqrt(mean(xr - xn).^2));
end

