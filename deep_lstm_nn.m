function [net_outputs, train_samples] = deep_lstm_nn(t, x, xn_train, xn_test, sample_length, result_length, predict_offset, samples_div, hiddenSizes, maxEpochs, hiddenType, snr_array)
    % Print options
    %fprintf("Samples: [%.2f:%.2f] Train sample div: %.2f\n", sample_length, result_length, samples_div);
    %fprintf("Network: Hidden: %u Epochs: %u Type: '%s'\n", hiddenSizes, maxEpochs, hiddenType);

    assert(hiddenType == "lstm" || hiddenType == "gru");

    % Create neural network
    numHiddenUnits = hiddenSizes;
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
        sequenceInputLayer(1)
        rnnBlock(numHiddenUnits, hiddenType, "tanh", 0.43, 2)
        fullyConnectedLayer(1)
        regressionLayer
    ];
    disp(layers);

    options = trainingOptions('adam', ... % sgdm, rmsprop, adam
        'MaxEpochs', maxEpochs, ...
        'SequenceLength', sample_length, ... % longest, shortest, <num>
        'GradientThreshold', 4.54, ...
        'Verbose', 0, ...
        'Plots', 'none', ... % 'training-progress', 'none'
        'InitialLearnRate',0.001, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...
        'LearnRateDropFactor',0.2, ...
        'Shuffle', 'never');

    % Additional training options:
    %
    %'MiniBatchSize', 64, ...
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
    [samples, results] = prepare_train_data(x, xn_train, sample_length, result_length, predict_offset, samples_div, snr_array);

    % Convert arrays to fit network inputs
    train_samples = samples.';
    train_results = results.';

    %fprintf("train samples:\n");
    %disp(size(train_samples));
    %disp(samples);
    %disp(size(train_results));
    %disp(results);

    % Train network
    net = trainNetwork(train_samples, train_results, layers, options);

    % Test network
    [test_samples, test_results] = prepare_train_data(x, xn_test, sample_length, result_length, predict_offset);
    net_outputs = test_network(net, test_samples, result_length, hiddenType);
end
