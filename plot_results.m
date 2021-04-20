function plot_results(name, t, xt, xr, xn, X, save_figure, t_skip)
    if ~exist('save_figure', 'var')
        save_figure = false;
    end

    if ~exist('t_skip', 'var')
        t_skip = 0;
    end

    %fprintf("plot_results: t_skip: %u\n", t_skip);
    %fprintf("t: "); disp(size(t));
    %fprintf("X: "); disp(size(X));
    
    ts = t;
    xrs = xr;

    % Skip some values from the end according to time skip value
    ts = ts(1 + t_skip:length(ts));
    xrs = xrs(1 + t_skip:length(xrs));

    % Align ts and xrs to X
    if length(ts) > length(X)
        diff = length(ts) - length(X) + 1;
        fprintf("Skip ts: %u\n", diff);

        ts = t(diff:length(ts));
        xrs = xr(diff:length(xrs));
    end

    assert(length(ts) == length(X));
    assert(length(xrs) == length(X));

    fig_nn = figure('name', name);
    tiledlayout(4, 1);

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
    ylim([0 1]);
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
    ylim([0 1]);
    hold off;

    % Calculate error for each time step
    [error, abs_error, mse_array, rmse_array] = calc_errors(xrs, X);

    % Calculate error values
    max_error = max(abs_error);
    mean_error = mean(abs_error);
    mse = mean(error.^2);
    rmse = sqrt(mse);

    fprintf("%s\n", name);
    fprintf("Mean error: %f\n", mean_error);
    fprintf("Max error:  %f\n", max_error);
    fprintf("MSE:        %f\n", mse);
    fprintf("RMSE:       %f\n", rmse);
    
    % Plot absolute error
    nexttile;
    plot(ts, abs_error);
    title(sprintf('Absolute error, mean: %.2f, max: %.2f', mean_error, max_error));
    xlabel('Time');
    ylabel('Absolute error');
    ylim([0 1]);

    % Plot MSE and RMSE on a single plot
    nexttile;
    hold on;
    plot(ts, mse_array);
    plot(ts, rmse_array);
    title(sprintf('Final MSE: %f RMSE: %f', mse, rmse));
    xlabel('Time');
    ylabel('MSE/RMSE');
    hold off;

    % Save plot image
    if save_figure
        date_str = datestr(datetime(), 'yyyymmdd_HHMMSS');
        str_name = sprintf('%s_date_%s', name, date_str);
        saveas(fig_nn, str_name + ".png");
    end
end
