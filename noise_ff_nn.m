function net_outputs = noise_ff_nn(t, x, xn_train, xn_test, sample_length, result_length, samples_div, hiddenSizes, maxEpochs, trainFcn)
    % Print options
    fprintf("Samples: [%.2f:%.2f] Train sample div: %.2f\n", sample_length, result_length, samples_div);
    fprintf("Network: Hidden: %u Train: '%s'\n", hiddenSizes, trainFcn);

    % Create neural network
    net = feedforwardnet(hiddenSizes, trainFcn);

    % Prepare train data set
    [samples, results] = prepare_train_data(x, xn_train, sample_length, result_length, 0, samples_div);
    samples_num = length(samples);

    % Transpose test arrays to fit network inputs
    samples = samples.';
    results = results.';

    % Configure network inputs:
    net = configure(net, samples, results);
    %fprintf("net.inputs: %d\n", net.inputs{1}.size);

    % Train network
    [net, tr] = train(net, samples, results);

    % Test network
    [test_samples, test_results] = prepare_train_data(x, xn_test, sample_length, result_length, 0);
    net_outputs = test_network(net, test_samples, result_length);

    % Assess the performance of the trained network.
    %
    % The default performance function is mean squared error.
    perf = perform(net, x, net_outputs.');
end
