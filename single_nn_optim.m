clear all; % Clear variables
close all; % Close plots
clc;       % Clear command window

% Initialize random number generator
rng(12345, 'combRecursive');

% Define input data
t = 1:0.1:10;

% Generate test data (real target position)
r = 0.01;
global snr;
snr = 5;

% Data set 1 (xr1, xr2)
w = 3 * pi;
phi = 0;
A = 0.5 + floor(t);

global xr;
global xn_train;
global xn_test; 

[xr, xn_train] = gen_sin(t, A, w, phi, r, snr);
xn_test = xn_train;

global result_length;
global samples_div;
result_length = 1;
samples_div = 1.5;

global snr_array;
snr_array = [snr snr snr];

% Define variables to optimize:
%
% See also:
% https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html
optimVars = [
    optimizableVariable('sequenceLength', [1 10], 'Type', 'integer')
    optimizableVariable('hiddenSize', [1 100], 'Type', 'integer')
    %optimizableVariable('type', ["lstm" "gru"]) % "lstm" "gru"
    optimizableVariable('actType', ["relu" "tanh"]) % "leakyrelu" "clippedrelu" "elu" "swish"
    optimizableVariable('numLayers', [1 3], 'Type', 'integer')
    optimizableVariable('dropout', [0 0.5])
    optimizableVariable('gradientThreshold', [0.1 100], 'Transform', 'log')
    optimizableVariable('initialLearnRate', [0.001 1], 'Transform', 'log')];

BayesObject = bayesopt(make_validation_fcn, optimVars,  ...
    'MaxObjectiveEvaluations', 100, ...
    'MaxTime', 14*60*60, ...
    'IsObjectiveDeterministic', false, ...
    'UseParallel', false);

% Get the best network
bestIdx = BayesObject.IndexOfMinimumTrace(end);
filename = BayesObject.UserDataTrace{bestIdx};

% Load network from file
fprintf("Loading file: %s\n", filename);
savedStruct = load(filename);

mean_error = savedStruct.mean_error;
net = savedStruct.net;
opt = savedStruct.optimVars;

disp(net);
disp(opt);

sample_length = 1;

[test_samples, test_results] = prepare_train_data(xr, xn_test, sample_length, result_length, 0);
X = test_network(net, test_samples, result_length, "lstm");

[xrs, X] = align_data(xr, X);

[error, abs_error, mse_array, rmse_array, max_error, mean_error, mse, rmse] = calc_errors(xrs, X);

fprintf("SL: %u HS: %u ME: %f MSE: %f\n", opt.sequenceLength, opt.hiddenSize, mean_error, mse);
plot_results("Best NN configuration", t, xr, xr, xn_test, X, false, 0, samples_div);

function layers = rnnBlock(hiddenSize, hiddenType, activation, numLayers, dropout)
    assert(hiddenType == "lstm" || hiddenType == "gru");
    assert(activation == "relu" || activation == "tanh" || activation == "none");

    if ~exist('hiddenType', 'var')
        hiddenType = "lstm"
    end

    if ~exist('numLayers', 'var')
        numLayers = 1;
    end

    if ~exist('dropout', 'var')
        dropout = 0;
    end

    % Determine main working layer type
    layer = [lstmLayer(hiddenSize)];

    if hiddenType == "gru"
        layer(end,:) = gruLayer(hiddenSize);
    end

    % Determine activation type
    if activation == "relu"
        layer(end + 1,:) = reluLayer;
    elseif activation == "tanh"
        layer(end + 1,:) = tanhLayer;
    end

    if (dropout > 0)
        layer(end + 1,:) = dropoutLayer(dropout);
    end

    % Copy layers 'numLayers' time
    layers = repmat(layer, numLayers, 1);
end

function ObjFcn = make_validation_fcn()
ObjFcn = @validation_fcn;
    function [mean_error, cons, filename] = validation_fcn(optimVars)
        % Define input data
        sample_length = optimVars.sequenceLength;

        % Create neural network
        maxEpochs = 100;

        layers = [ ...
            sequenceInputLayer(1)
            rnnBlock(optimVars.hiddenSize, "lstm", optimVars.actType, optimVars.numLayers, optimVars.dropout)
            fullyConnectedLayer(1)
            regressionLayer
        ];
        %disp(layers);

        options = trainingOptions('adam', ... % sgdm, rmsprop, adam
            'MaxEpochs', maxEpochs, ...
            'SequenceLength', sample_length, ... % longest, shortest, <num>
            'GradientThreshold', optimVars.gradientThreshold, ...
            'InitialLearnRate', optimVars.initialLearnRate, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod', 125, ...
            'LearnRateDropFactor', 0.2, ...
            'Shuffle', 'never', ...
            'Verbose', 0, ...
            'Plots', 'none'); % 'training-progress', 'none'

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
        global xr;
        global xn_train;
        global xn_test;
        global result_length;
        global samples_div;
        global snr_array;
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

        [xrs, X]= align_data(xr, X);

        [error, abs_error, mse_array, rmse_array, max_error, mean_error, mse, rmse] = calc_errors(xrs, X);

        fprintf("SL: %u HS: %u ME: %f MSE: %f\n", sample_length, optimVars.hiddenSize, mean_error, mse);

        % Save trained network to file
        filename = "bayesopt_" + num2str(mean_error) + ".mat";
        fprintf("Saving to file: %s\n", filename);
        save(filename, 'net', 'mean_error', 'optimVars');

        cons = [];
    end
end
