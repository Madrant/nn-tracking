function net_outputs = noise_lstm_nn(t, x, xn_train, xn_test, sample_length, result_length, samples_div, hiddenSizes, maxEpochs, hiddenType, snr_array)
    % Print options
    fprintf("Samples: [%.2f:%.2f] Train sample div: %.2f\n", sample_length, result_length, samples_div);
    fprintf("Network: Hidden: %u Epochs: %u Type: '%s'\n", hiddenSizes, maxEpochs, hiddenType);

    assert(hiddenType == "lstm" || hiddenType == "gru");

    % Prepare train data set
    [samples, results] = prepare_train_data(x, xn_train, sample_length, result_length, 0, samples_div);
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
    test_step = 1;

    real_data = zeros(samples_num, sample_length);
    measurements = zeros(samples_num, sample_length);
    net_outputs = zeros(samples_num, result_length);

    for n = 1 : test_step: samples_num
        real = x(n: n + sample_length - 1);
        measurement = xn_test(n: n + sample_length - 1);

        %net_output = predict(net, measurement.');
        [net, net_output] = predictAndUpdateState(net, measurement.');

        real_data(n,:) = real;
        measurements(n,:) = measurement;

        % Skip first network output to align input data
        % with network prediction
        if n == 1
            %continue;
        end

        %disp(measurement);
        %disp(net_output);

        net_outputs(n,:) = net_output(1);
    end

    %fprintf("xn_test: "); disp(size(xn_test));
    %fprintf("outputs: "); disp(size(net_outputs));
    
    % Convert column to row
    net_outputs = net_outputs.';
end
