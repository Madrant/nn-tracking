function net_outputs = ff_nn(t, x, xn, sample_length, result_length, samples_div, hiddenSizes, trainFcn)
    % Print options
    fprintf("Samples: [%f:%f] Train sample div: %f\n", sample_length, result_length, samples_div);
    fprintf("Network: Hidden: %f Train: '%s'\n", hiddenSizes, trainFcn);

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

    % Assess the performance of the trained network.
    %
    % The default performance function is mean squared error.
    perf = perform(net, x, net_outputs);

    % Convert column to row
    net_outputs = net_outputs.';
end