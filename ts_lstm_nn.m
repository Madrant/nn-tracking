% Predict time series using LSTM neural network
%
% See also:
% https://www.mathworks.com/help/deeplearning/ug/time-series-forecasting-using-deep-learning.html
function net_outputs = ts_lstm_nn(t, x, xn, sample_length, result_length, samples_div, hiddenSizes, maxEpochs)
    % Print options
    fprintf("Samples: [%.2f:%.2f] Train sample div: %.2f\n", sample_length, result_length, samples_div);

    % Prepare train data set
    samples_num = length(x) - (sample_length + result_length - 1);
    train_samples_num = round(length(x) / samples_div) - (sample_length + result_length - 1);

    samples = zeros(train_samples_num, sample_length);
    results = zeros(train_samples_num, result_length);

    for n = 1 : train_samples_num
        samples(n,:) = x(n:n + sample_length - 1);
        results(n,:) = x(n + sample_length:n + sample_length + result_length - 1);
    end

    % Transpose test arrays to fit network inputs
    samples = samples.';
    results = results.';

    % Create neural network
    numHiddenUnits = hiddenSizes;
    maxEpochs = maxEpochs;

    layers = [ ...
        sequenceInputLayer(sample_length)
        lstmLayer(numHiddenUnits, 'OutputMode', 'sequence') % sequence, last
        fullyConnectedLayer(1)
        regressionLayer
    ];

    options = trainingOptions('sgdm', ... % sgdm, rmsprop, adam
        'MaxEpochs', maxEpochs, ...
        'GradientThreshold', 1, ...
        'Verbose', 0, ...
        'Plots', 'training-progress');

    % Additional training options:
    %
    % 'Momentum', 0.9, ... % only for sgdm
    %'InitialLearnRate', 0.005, ...
    %'LearnRateSchedule', 'piecewise', ...
    %'LearnRateDropPeriod', 125, ...
    %'LearnRateDropFactor', 0.2, ...        
    %'Shuffle', 'never', ... % once, never, every-epoch

    % Train network
    net = trainNetwork(samples, results, layers, options);

    % Test network
    test_step = 1;

    real_data = zeros(samples_num, result_length);
    measurements = zeros(samples_num, result_length);
    net_outputs = zeros(samples_num, result_length);

    for n = 1 : test_step: samples_num
        test_sample = xn(n:n + sample_length - 1);

        real = x(n + sample_length:n + sample_length + result_length - 1);
        measurement = xn(n + sample_length:n + sample_length + result_length - 1);

        %net_output = predict(net, test_sample.').';
        [net, net_output] = predictAndUpdateState(net, test_sample.');

        real_data(n,:) = real;
        measurements(n,:) = measurement;
        net_outputs(n,:) = net_output.';
    end

    % Convert column to row
    net_outputs = net_outputs.';
end
