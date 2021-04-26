function [net_outputs, train_samples] = noise_lstm_nn(t, x, xn_train, xn_test, sample_length, result_length, samples_div, hiddenSizes, maxEpochs, hiddenType, snr_array)
    % Print options
    fprintf("Samples: [%.2f:%.2f] Train sample div: %.2f\n", sample_length, result_length, samples_div);
    fprintf("Network: Hidden: %u Epochs: %u Type: '%s'\n", hiddenSizes, maxEpochs, hiddenType);

    assert(hiddenType == "lstm" || hiddenType == "gru");

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

    % LSTM and GRU layer parameters
    %
    % See also:
    % https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.grulayer.html
    % https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.lstmlayer.html
    %
    % 'InputWeightsInitializer', 'zeros', ...
    % 'RecurrentWeightsInitializer', 'zeros', ...
    % 'BiasInitializer', 'ones'

    layers = [ ...
        sequenceInputLayer(1)
        mainLayer
        fullyConnectedLayer(1)
        regressionLayer
    ];

    % https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html
    options = trainingOptions('adam', ... % sgdm, rmsprop, adam
        'MaxEpochs', maxEpochs, ...
        'SequenceLength', sample_length, ...
        'GradientThreshold', 1, ...
        'Verbose', 0, ...
        'Plots', 'none', ... % 'training-progress', 'none'
        'InitialLearnRate',0.005, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...
        'LearnRateDropFactor',0.2, ...
        'Shuffle', 'once');

    % Additional training options:
    %
    % 'MiniBatchSize', 64, ...
    %'GradientThreshold', 1, ...
    %'Momentum', 0.9, ... % only for sgdm
    %'InitialLearnRate', 0.005, ...
    %'LearnRateSchedule', 'piecewise', ...
    %'LearnRateDropPeriod', 125, ...
    %'LearnRateDropFactor', 0.2, ...        
    %'Shuffle', 'never', ... % once, never, every-epoch

    % Prepare train data
    %
    % sample_length = sample_length;
    % [samples, results] = prepare_train_data(x, xn_train, sample_length, result_length, 0, samples_div, snr_array);

    % train_samples = samples; train_results = results;
    sample_length = 1;
    [samples, results] = prepare_train_data(x, xn_train, sample_length, result_length, 0, samples_div, snr_array);

    % Convert arrays to fit network inputs
    train_samples = samples.';
    train_results = results.';

    fprintf("train samples:\n");
    disp(size(train_samples));
    %disp(samples);
    disp(size(train_results));
    %disp(results);

    % Train network
    net = trainNetwork(train_samples, train_results, layers, options);

    % Test network
    [test_samples, test_results] = prepare_train_data(x, xn_test, sample_length, result_length, 0);
    net_outputs = test_network(net, test_samples, result_length, hiddenType);
end
