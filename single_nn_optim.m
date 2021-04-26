clear all; % Clear variables
close all; % Close plots
clc;       % Clear command window

% Initialize random number generator
rng(12345, 'combRecursive');

% Define variables to optimize:
optimVars = [
    optimizableVariable('SequenceLength',   [1 10], 'Type', 'integer')
    optimizableVariable('hiddenSize', [1 100], 'Type', 'integer')
    optimizableVariable('InitialLearnRate', [0.005 1], 'Transform', 'log')];

BayesObject = bayesopt(make_validation_fcn, optimVars,  ...
    'MaxObjectiveEvaluations', 100, ...
    'MaxTime', 14*60*60, ...
    'IsObjectiveDeterministic', false, ...
    'UseParallel', false);

function ObjFcn = make_validation_fcn(xr1)
ObjFcn = @validation_fcn;
    function [mean_error, cons] = validation_fcn(optimVars, xr1)
        % Define input data
        t = 1:0.1:10;

        % Generate test data (real target position)
        r = 0.01;
        snr = 5;

        % Data set 1 (xr1, xr2)
        w = 3 * pi;
        phi = 0;
        A = 0.5 + floor(t);

        [xr, xn_train] = gen_sin(t, A, w, phi, r, snr);
        xn_test = xn_train;

        sample_length = optimVars.SequenceLength;
        result_length = 1;

        samples_div = 1.5;
        snr_array = [snr snr snr];
        
        % Create neural network
        maxEpochs = 100;

        layers = [ ...
            sequenceInputLayer(1)
            lstmLayer(optimVars.hiddenSize)
            fullyConnectedLayer(1)
            regressionLayer
        ];
        %disp(layers);

        options = trainingOptions('adam', ... % sgdm, rmsprop, adam
            'MaxEpochs', maxEpochs, ...
            'SequenceLength', sample_length, ... % longest, shortest, <num>
            'GradientThreshold', 1, ...
            'Verbose', 0, ...
            'Plots', 'none', ... % 'training-progress', 'none'
            'InitialLearnRate', 0.005, ...
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
        [samples, results] = prepare_train_data(xr, xn_train, sample_length, result_length, 0, samples_div, snr_array);

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
        [test_samples, test_results] = prepare_train_data(xr, xn_test, sample_length, result_length, 0);
        X = test_network(net, test_samples, result_length, "lstm");

        [ts, X] = align_data(t, X);
        [xrs, X]= align_data(xr, X);

        [error, abs_error, mse_array, rmse_array, max_error, mean_error, mse, rmse] = calc_errors(xrs, X);

        fprintf("SL: %u HS: %u ME: %f MSE: %f\n", sample_length, optimVars.hiddenSize, mean_error, mse);

        cons = [];
    end
end
