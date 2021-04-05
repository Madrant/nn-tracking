function net_outputs = ts_lstm_nn(t, x, xn, sample_length, result_length, samples_div)
    % Print options
    fprintf("Samples: [%f:%f] Train sample div: %f\n", sample_length, result_length, samples_div);

    % Initial weights are random unless we initialize random number generator
    % rng(rnd_seed, 'combRecursive');

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

    % Create neural network
    numHiddenUnits = 10;
    maxEpochs = 50;
    
    layers = [ ...
        sequenceInputLayer(sample_length)
        lstmLayer(numHiddenUnits, 'OutputMode', 'sequence') % sequence, last
        fullyConnectedLayer(1)
        regressionLayer
    ];

    options = trainingOptions('sgdm', ... % sgdm, rmsprop, adam
        'ExecutionEnvironment','cpu', ...
        'GradientThreshold', 1, ...
        'Momentum', 0.9, ...
        'MaxEpochs', maxEpochs, ...
        'Shuffle', 'never', ... % once, never, every-epoch
        'Verbose', 0, ...
        'Plots', 'training-progress');

    % Train network
    disp(size(samples));
    disp(size(results));
    net = trainNetwork(samples, (results), layers, options);

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

        net_output = predict(net, test_sample.').';

        real_data(n,:) = real;
        measurements(n,:) = measurement;
        net_outputs(n,:) = net_output;
    end

    % Convert column to row
    net_outputs = net_outputs.';
end
