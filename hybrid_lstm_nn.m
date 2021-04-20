function [net1_outputs, net2_outputs] = hybrid_lstm_nn(t, x, xn_train, xn_test, sl_array, rl_array, samples_div, hs_array, maxEpochs, hiddenType, snr_array)
    % Print options
    fprintf("Samples: [%.2f:%.2f] Train sample div: %.2f\n", sl_array(2), rl_array(2), samples_div);
    fprintf("Network: Hidden: %u:%u Epochs: %u Type: '%s'\n", hs_array(1), hs_array(2), maxEpochs, hiddenType);

    assert(hiddenType == "lstm" || hiddenType == "gru");

    % Configure networks
    net1_sample_length = sl_array(1);
    net1_result_length = rl_array(1);

    net2_sample_length = sl_array(2);
    net2_result_length = rl_array(2);

    if ~exist('maxEpochs', 'var')
        maxEpochs = 100;
    end

    % Select hidden layer type
    if hiddenType == "lstm"
        net1_mainLayer = lstmLayer(hs_array(1));
        net2_mainLayer = lstmLayer(hs_array(2));
    end
    if hiddenType == "gru"
        net1_mainLayer = gruLayer(hs_array(1));
        net2_mainLayer = gruLayer(hs_array(2));
    end

    net1_layers = [ ...
        sequenceInputLayer(net1_sample_length)
        net1_mainLayer
        fullyConnectedLayer(net1_result_length)
        regressionLayer
    ];

    net2_layers = [ ...
        sequenceInputLayer(net2_sample_length)
        net2_mainLayer
        fullyConnectedLayer(net2_result_length)
        regressionLayer
    ];

    net1_options = trainingOptions('sgdm', ... % sgdm, rmsprop, adam
        'MaxEpochs', maxEpochs, ...
        'MiniBatchSize', net1_sample_length, ...
        'GradientThreshold', 1, ...
        'Verbose', 0, ...
        'Plots', 'none', ... % 'training-progress', 'none'
        'InitialLearnRate',0.005, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...
        'LearnRateDropFactor',0.2, ...
        'Shuffle', 'never');

    net2_options = trainingOptions('sgdm', ... % sgdm, rmsprop, adam
        'MaxEpochs', maxEpochs, ...
        'MiniBatchSize', net2_sample_length, ...
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
    
    % Prepare train data set for net1
    [net1_samples, net1_results] = prepare_train_data(x, xn_train, net1_sample_length, net1_result_length, 0, samples_div, snr_array);

    % Train net1
    net1 = trainNetwork(net1_samples.', net1_results.', net1_layers, net1_options);

    % Get net1 output to train net2
    [net1_test_samples, net1_test_results] = prepare_train_data(x, xn_train, net1_sample_length, net1_result_length, 0, 1, snr_array);
    net1_outputs = test_network(net1, net1_test_samples, net1_result_length, hiddenType);

    net1_outputs = net1_outputs.';

    fprintf("net1 output:\n")
    fprintf("inputs:  "); disp(size(net1_test_samples));
    fprintf("outputs: "); disp(size(net1_outputs));

    % Generate train data for second network
    loops = length(snr_array) + 1;

    net2_test_samples = [];
    net2_test_results = [];

    for n = 1:loops
        from = length(net1_outputs) / loops * (n - 1) + 1;
        to = length(net1_outputs) / loops * n;

        [test_samples, test_results] = prepare_train_data(...
            x, net1_outputs(from:to).', net2_sample_length, net2_result_length);

        net2_test_samples = [net2_test_samples; test_samples];
        net2_test_results = [net2_test_results; test_results];
    end

    % Add reference signal to net2 train data
    if false
        [test_samples, test_results] = prepare_train_data(...
                x, x, net2_sample_length, net2_result_length, 0);

        net2_test_samples = [net2_test_samples; test_samples];
        net2_test_results = [net2_test_results; test_results];
    end

    fprintf("net2 train data:\n");
    fprintf("inputs:  "); disp(size(net2_test_samples));
    fprintf("outputs: "); disp(size(net2_test_results));

    % Train second network
    net2 = trainNetwork(net2_test_samples.', net2_test_results.', net2_layers, net2_options);

    % Process data by using net1 and net2 together
    [test_samples, test_results] = prepare_train_data(x, xn_test, net1_sample_length, net1_result_length, 0);

    % Process noise by net1
    net1_outputs = test_network(net1, test_samples, net1_result_length, hiddenType);

    [test_samples, test_results] = prepare_train_data(x, net1_outputs, net2_sample_length, net2_result_length);

    % Approximate net1 output by net2
    net2_outputs = test_network(net2, test_samples, net2_result_length, hiddenType);
end
