function plot_results(name, t, xt, xr, xn, t_test, xr_test, xn_test, X, save_figure, samples_div, t_skip)
    if ~exist('save_figure', 'var')
        save_figure = false;
    end

    if ~exist('t_skip', 'var')
        t_skip = 0;
    end

    if ~exist('samples_div', 'var')
        samples_div = 1;
    end

    assert(samples_div ~= 0);

    ts = t;
    xrs = xr;

    % Skip some values from the end according to time skip value
    ts = ts(1 + t_skip:length(ts));
    xrs = xrs(1 + t_skip:length(xrs));

    % Align ts and xrs to X
    [ts, X] = align_data(ts, X);
    [xrs, X]= align_data(xrs, X);

    assert(length(ts) == length(X));
    assert(length(xrs) == length(X));

    fig_nn = figure('name', name);
    tiledlayout(4, 1);

    % Plot input data
    %
    % calculate train_threshold
    t_div = t(round(length(t) / samples_div));

    nexttile;
    hold on;
    plot(t, xt);
    plot(t, xr);
    plot(t, xn, '-x'); % Plot measurements
    plot([t_div t_div], [0 1]);
    grid on;
    title('Model and Measurements');
    xlabel('Time');
    ylabel('Data');
    legend('Model', 'Real data', 'Measurements');
    ylim([0 1]);
    hold off;

    % Plot network output
    nexttile;
    hold on;
    plot(t_test, xr_test);
    plot(t_test, xn_test, '-x');
    plot(t_test, X, '-o');
    plot([t_div t_div], [0 1]);
    grid on;
    title('Filter output');
    xlabel('Time');
    ylabel('Data');
    legend('Real data', 'Measurements', 'Filter output');
    ylim([0 1]);
    hold off;

    % Calculate error for each time step
    [error, abs_error, mse_array, rmse_array, max_error, mean_error, mse, rmse] = calc_errors(xr_test, X);

    fprintf("%s\n", name);
    fprintf("Mean error: %f\n", mean_error);
    fprintf("Max error:  %f\n", max_error);
    fprintf("MSE:        %f\n", mse);
    fprintf("RMSE:       %f\n", rmse);
    
    % Plot absolute error
    nexttile;
    plot(t_test, abs_error);
    grid on;
    title(sprintf('Absolute error, mean: %.2f, max: %.2f', mean_error, max_error));
    xlabel('Time');
    ylabel('Absolute error, value');
    ylim([0 1]);
    yticks([0 0.1 0.2 0.3 0.5 0.75 1]);

    % Plot MSE and RMSE on a single plot
    nexttile;
    hold on;
    plot(t_test, mse_array);
    plot(t_test, rmse_array);
    grid on;
    title(sprintf('Final MSE: %f RMSE: %f', mse, rmse));
    xlabel('Time');
    ylabel('MSE / RMSE');
    hold off;

    % Save plot image
    if save_figure
        date_str = datestr(datetime(), 'yyyymmdd_HHMMSS');
        str_name = sprintf('%s_date_%s', name, date_str);
        saveas(fig_nn, str_name + ".png");
    end
end
