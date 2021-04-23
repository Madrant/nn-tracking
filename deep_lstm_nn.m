function net_outputs = deep_lstm_nn(t, x, xn_train, xn_test, sample_length, result_length, samples_div, hiddenSizes, maxEpochs, hiddenType, snr_array)
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
    numHiddenUnits = 10;
    maxEpochs = maxEpochs;

    % Select hidden layer type
    if hiddenType == "lstm"
        mainLayer = lstmLayer(numHiddenUnits, ...
            'OutputMode', 'sequence', ... % dequence (default), last
            'StateActivationFunction', 'tanh', ... % tanh (default), softsign
            'GateActivationFunction', 'sigmoid', ... % sigmoid (default), hard-sigmoid
            'BiasInitializer', 'unit-forget-gate'); % unit-forget-gate (default), narrow-normal, ones
    elseif hiddenType == "gru"
        mainLayer = gruLayer(numHiddenUnits, ...
            'OutputMode', 'sequence', ... % dequence (default), last
            'ResetGateMode', 'after-multiplication', ... % after-multiplication (default) , before-multiplication, recurrent-bias-after-multiplication
            'StateActivationFunction', 'softsigh', ... % tanh (default), softsign
            'GateActivationFunction', 'sigmoid', ... % sigmoid (default), hard-sigmoid
            'BiasInitializer', 'unit-forget-gate'); % unit-forget-gate (default), narrow-normal, ones);
    end

    layers = [ ...
        sequenceInputLayer(sample_length)
        lstmLayer(10)
        reluLayer
        lstmLayer(10)
        reluLayer
        fullyConnectedLayer(1)
        %dropoutLayer(0.2)
        regressionLayer
    ];
    disp(layers);

    options = trainingOptions('adam', ... % sgdm, rmsprop, adam
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