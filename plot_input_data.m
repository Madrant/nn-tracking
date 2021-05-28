function plot_input_data(t, xr_train, xn_train, xr_test, xn_test)
    fig_input = figure('Name', "Input data");
    tiledlayout(2, 1);

    nexttile;
    hold on;
    grid on;
    plot(t, xr_train, '-d');
    plot(t, xn_train, '-x');
    legend("Train Data", "Train Measurements");
    hold off;

    nexttile;
    hold on;
    plot(t, xr_test, '-d');
    plot(t, xn_test, '-x');
    legend("Test Data", "Test Measurements");
    hold off;
end

