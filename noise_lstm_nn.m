function net_outputs = noise_lstm_nn(t, x, xn_train, xn_test, sample_length, result_length, samples_div, hiddenSizes, maxEpochs, hiddenType, snr_array)
    % Print options
    fprintf("Samples: [%.2f:%.2f] Train sample div: %.2f\n", sample_length, result_length, samples_div);
    fprintf("Network: Hidden: %u Epochs: %u Type: '%s'\n", hiddenSizes, maxEpochs, hiddenType);

    assert(hiddenType == "lstm" || hiddenType == "gru");

    % Prepare train data set
    [samples, results] = prepare_train_data(x, xn_train, sample_length, result_length, 0, samples_div, snr_array);
    samples_num = length(samples);

    %fprintf("train samples:\n");
    %disp(size(samples));
    %disp(size(results));

    % Transpose test arrays to fit network inputs
    samples = samples.';
    results = results.';

    % Create neural network
    numHiddenUnits = hiddenSizes;
    maxEpochs = maxEpochs;

    % Select hidden layer type
    if hiddenType == "lstm"
        mainLayer = lstmLayer(numHiddenUnits);
    end
    if hiddenType == "gru"
        mainLayer = gruLayer(numHiddenUnits);
    end

    layers = [ ...
        sequenceInputLayer(sample_length)
        mainLayer
        fullyConnectedLayer(1)
        regressionLayer
    ];

    options = trainingOptions('sgdm', ... % sgdm, rmsprop, adam
        'MaxEpochs', maxEpochs, ...
        'MiniBatchSize', sample_length, ...
        'GradientThreshold', 1, ...
        'Verbose', 0, ...
        'Plots', 'none', ... % 'training-progress', 'none'
        'InitialLearnRate',0.005, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...
        'LearnRateDropFactor',0.2, ...
        'Shuffle', 'never');

    % Additional training options:
    %
    %'GradientThreshold', 1, ...
    %'Momentum', 0.9, ... % only for sgdm
    %'InitialLearnRate', 0.005, ...
    %'LearnRateSchedule', 'piecewise', ...
    %'LearnRateDropPeriod', 125, ...
    %'LearnRateDropFactor', 0.2, ...        
    %'Shuffle', 'never', ... % once, never, every-epoch

    % Train network
    net = trainNetwork(samples, results, layers, options);

    % Test network
    [test_samples, test_results] = prepare_train_data(x, xn_test, sample_length, result_length, 0);
    net_outputs = test_network(net, test_samples, result_length, hiddenType);
end
