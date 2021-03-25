function [max_error, mse, rmse, tr] = time_series_forecasting(t, x, xn, sample_length, result_length, samples_div, hiddenSizes, trainFcn, saveFigure)
    % Print options
    fprintf("Samples: [%f:%f] Train sample div: %f\n", sample_length, result_length, samples_div);
    fprintf("Network: Hidden: %f Train: '%s'\n", hiddenSizes, trainFcn);

    fprintf("Measurements Max error: %f\n", max(abs(x - xn)));
    fprintf("Measurements MSE:       %f\n", mean(x - xn).^2);
    fprintf("Measurements RMSE:      %f\n", sqrt(mean(x - xn).^2));

    % Setup plot layout
    fig_nn = figure('name', sprintf('NN Tracking: trainFcn: %s', trainFcn));
    tiledlayout(5, 1);

    % Plot input data
    nexttile;
    hold on;
    plot(t, x);
    plot(t, xn, 'x'); % Plot measurements
    title('Measurements');
    xlabel('Time');
    ylabel('Data');
    legend('Training sample', 'Test sample');
    hold off;

    %waitforbuttonpress;

    % Initial weights are random unless we initialize random number generator
    % rng(rnd_seed, 'combRecursive');

    % Create neural network
    net = feedforwardnet(hiddenSizes, trainFcn);

    % Prepare train data set
    samples_num = round(length(x) / samples_div) - (sample_length + result_length - 1);

    samples = zeros(samples_num, sample_length);
    results = zeros(samples_num, result_length);

    for n = 1 : samples_num
        samples(n,:) = x(n:n + sample_length - 1);
        results(n,:) = x(n + sample_length:n + sample_length + result_length - 1);
    end

    % Plot training sample length
    % plot(t(1:n), x(1:n));

    % Print generated test data
    % fprintf("samples: "); disp(samples);
    % fprintf("results: "); disp(results);

    % Transpose test arrays to fit network inputs
    samples = samples.';
    results = results.';

    % Configure network inputs:
    net = configure(net, samples, results);
    fprintf("net.inputs: %d\n", net.inputs{1}.size);

    % Train network
    [net, tr] = train(net, samples, results);

    % Test network
    test_step = 1;

    samples_num = round(length(x) / test_step) - (sample_length + result_length - 1);

    real_data = zeros(samples_num, result_length);
    measurements = zeros(samples_num, result_length);
    net_outputs = zeros(samples_num, result_length);

    for n = 1 : test_step: length(xn) - (sample_length + result_length - 1)
        test_sample = xn(n:n + sample_length - 1);

        real = x(n + sample_length:n + sample_length + result_length - 1);
        measurement = xn(n + sample_length:n + sample_length + result_length - 1);

        net_output = net(test_sample.').';

        real_data(n,:) = real;
        measurements(n,:) = measurement;
        net_outputs(n,:) = net_output;

        % fprintf("test_sample: "); disp(test_sample);
        % fprintf("net_output: "); disp(net_output);
        % fprintf("measurement: "); disp(measurement);

        % Plot network result
        % sample_time = t(n):time_step:t(n + length(test_sample) - 1);
        % result_time = t(n + sample_length):time_step:t(n + sample_length + length(net_output) - 1);

        % plot(sample_time, test_sample, 'x'); waitforbuttonpress;
        % plot(result_time, net_output, 'o'); waitforbuttonpress;
    end

    % Plot network output
    nexttile;
    hold on;
    plot(t(sample_length + 1:length(t)), measurements);
    plot(t(sample_length + 1:length(t)), net_outputs, 'o');
    plot(t(sample_length + 1:length(t)), real_data, 'x');
    title('Network output');
    xlabel('Time');
    ylabel('Data');
    legend('Test sample', 'Network output', 'Real data');
    hold off;

    % Assess the performance of the trained network.
    %
    % The default performance function is mean squared error.
    perf = perform(net, x, net_outputs);

    % Calculate absolute errors
    error = (measurements - net_outputs);
    abs_error = abs(error);
    max_error = max(abs_error);
    mean_error = mean(abs_error);
    mse = mean(error.^2);
    rmse = sqrt(mse);

    % Calculate MSE for time series
    mse_array = zeros(length(error));
    rmse_array = zeros(length(error));
    
    for n=1:length(error)
        mse = mean(error(1:n).^2);
        mse_array(n) = mse;
        rmse_array(n) = sqrt(mse);
    end

    fprintf("Network performance (MSE by defaults): "); disp(perf);
    fprintf("Max error: %f\n", max_error);
    fprintf("Mean error: %f\n", mean_error);
    fprintf("MSE:       %f\n", mse);
    fprintf("RMSE:      %f\n", rmse);

    % Plot absolute error
    nexttile;
    plot(t(sample_length + 1:length(t)), abs_error);
    title(sprintf('Absolute error, maximum: %.2f, mean: %.2f', max_error, mean_error));
    xlabel('Time');
    ylabel('Absolute error');

    % Plot MSE
    nexttile;
    plot(t(sample_length + 1:length(t)), mse_array);
    title(sprintf('Final MSE: %f', mse));
    xlabel('Time');
    ylabel('MSE');
    
    % Plot RMSE
    nexttile;
    plot(t(sample_length + 1:length(t)), rmse_array);
    title(sprintf('Final RMSE: %f', rmse));
    xlabel('Time');
    ylabel('RMSE');
    
    % Save plot image
    if saveFigure
        date_str = datestr(datetime(), 'yyyymmdd_HHMMSS');
        str_name = sprintf('nn_tracking_func_%s_hs_%u_date_%s', trainFcn, hiddenSizes, date_str);
        saveas(fig_nn, str_name + ".png");
    end
end
