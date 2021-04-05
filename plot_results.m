function plot_results(name, t, xt, xr, xn, X, save_figure, t_skip)
    % Skip some values from the start according to time skip value
    ts = t(t_skip + 1:length(t));
    xrs = xr(t_skip + 1: length(xr));

    fig_nn = figure('name', name);
    tiledlayout(5, 1);

    % Plot input data
    nexttile;
    hold on;
    plot(t, xt);
    plot(t, xr);
    plot(t, xn, '-x'); % Plot measurements
    title('Model and Measurements');
    xlabel('Time');
    ylabel('Data');
    legend('Model', 'Real data', 'Measurements');
    hold off;

    % Plot network output
    nexttile;
    hold on;
    plot(t, xr);
    plot(t, xn, '-x');
    plot(ts, X, '-o');
    title('Filter output');
    xlabel('Time');
    ylabel('Data');
    legend('Real data', 'Measurements', 'Filter output');
    hold off;

    % Calculate error for each time step
    [error, abs_error, mse_array, rmse_array] = calc_errors(xrs, X);

    % Calculate error values
    max_error = max(abs_error);
    mean_error = mean(abs_error);
    mse = mean(error.^2);
    rmse = sqrt(mse);

    fprintf("%s\n", name);
    fprintf("Max error:  %f\n", max_error);
    fprintf("Mean error: %f\n", mean_error);
    fprintf("MSE:        %f\n", mse);
    fprintf("RMSE:       %f\n", rmse);
    
    % Plot absolute error
    nexttile;
    plot(ts, abs_error);
    title(sprintf('Absolute error, maximum: %.2f, mean: %.2f', max_error, mean_error));
    xlabel('Time');
    ylabel('Absolute error');

    % Plot MSE
    nexttile;
    plot(ts, mse_array);
    title(sprintf('Final MSE: %f', mse));
    xlabel('Time');
    ylabel('MSE');

    % Plot RMSE
    nexttile;
    plot(ts, rmse_array);
    title(sprintf('Final RMSE: %f', rmse));
    xlabel('Time');
    ylabel('RMSE');

    % Save plot image
    if save_figure
        date_str = datestr(datetime(), 'yyyymmdd_HHMMSS');
        str_name = sprintf('%s_date_%s', name, date_str);
        saveas(fig_nn, str_name + ".png");
    end
end
