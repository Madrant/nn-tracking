function net_outputs = noise_ff_nn(t, x, xn, sample_length, result_length, samples_div, hiddenSizes, maxEpochs, trainFcn)
    % Print options
    fprintf("Samples: [%.2f:%.2f] Train sample div: %.2f\n", sample_length, result_length, samples_div);
    fprintf("Network: Hidden: %u Train: '%s'\n", hiddenSizes, trainFcn);

    % Create neural network
    net = feedforwardnet(hiddenSizes, trainFcn);

    % Prepare train data set
    samples_num = length(x) - (sample_length + result_length - 1);
    train_samples_num = round(length(x) / samples_div) - (sample_length + result_length - 1);

    snr_values = [];
    loops = length(snr_values) + 1;

    samples = zeros(train_samples_num * loops, sample_length);
    results = zeros(train_samples_num * loops, result_length);

    for loop = 1: loops
        if loop > length(snr_values)
            xnt = xn;
        else
            snr = snr_values(loop);
            xnt = awgn(x, snr, 'measured');
        end

        for n = 1 : train_samples_num
            samples(n + (train_samples_num * (loop - 1)),:) = xnt(n: n + sample_length - 1);
            results(n + (train_samples_num * (loop - 1)),:) = x(n + sample_length - 1:  n + sample_length + result_length - 2);
        end
    end

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

    real_data = zeros(samples_num, sample_length);
    measurements = zeros(samples_num, sample_length);
    net_outputs = zeros(samples_num, result_length);

    for n = 1 : test_step: samples_num
        real = x(n: n + sample_length - 1);
        measurement = xn(n: n + sample_length - 1);

        net_output = net(measurement.').';

        real_data(n,:) = real;
        measurements(n,:) = measurement;
        net_outputs(n,:) = net_output(1);
    end

    % Assess the performance of the trained network.
    %
    % The default performance function is mean squared error.
    perf = perform(net, x, net_outputs);

    % Convert column to row
    net_outputs = net_outputs.';
end
