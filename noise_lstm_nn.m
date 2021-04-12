function net_outputs = noise_lstm_nn(t, x, xn_train, xn_test, sample_length, result_length, samples_div)
    % Print options
    fprintf("Samples: [%u:%u]\n", sample_length, result_length);

    % Prepare train data set
    samples_num = length(x) - (sample_length + result_length - 1);
    train_samples_num = round(length(x) / samples_div) - (sample_length + result_length - 1);

    % Setup SNR for additional train data sets
    snr_values = [];
    loops = length(snr_values) + 1;

    samples = zeros(train_samples_num * loops, sample_length);
    results = zeros(train_samples_num * loops, result_length);

    for loop = 1: loops
        % Use real measurements
        if loop > length(snr_values)
            xnt = xn_train;
        else
            % Generate alternate noised measurements
            snr = snr_values(loop);
            xnt = awgn(x, snr, 'measured');
        end

        % Save additional datasets with the true one
        for n = 1 : train_samples_num
            samples(n + (train_samples_num * (loop - 1)),:) = xnt(n: n + sample_length - 1);
            results(n + (train_samples_num * (loop - 1)),:) = x(n + sample_length - 1:  n + sample_length + result_length - 2);
        end
    end

    disp(size(samples));
    disp(size(results));
    
    % Transpose test arrays to fit network inputs
    samples = samples.';
    results = results.';

        % Create neural network
    numHiddenUnits = 10;
    maxEpochs = 100;

    layers = [ ...
        sequenceInputLayer(sample_length)
        gruLayer(numHiddenUnits)
        %lstmLayer(numHiddenUnits)
        %lstmLayer(numHiddenUnits, 'OutputMode', 'sequence') % sequence, last
        %bilstmLayer(numHiddenUnits)
        fullyConnectedLayer(1)
        regressionLayer
    ];

    options = trainingOptions('sgdm', ... % sgdm, rmsprop, adam
        'MaxEpochs', maxEpochs, ...
        'GradientThreshold', 1, ...
        'Verbose', 0, ...
        'Plots', 'training-progress', ...
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
            continue;
        end

        net_outputs(n - 1,:) = net_output(1);
    end

    % Convert column to row
    net_outputs = net_outputs.';
end
