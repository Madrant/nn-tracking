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
snr = 10;

% Data set 1 (xr1, xr2)
w = 3 * pi;
phi = 0;
A = 5;

[xr1, xn1] = gen_sin(t, A, w, phi, r, snr);

% Data set 2 (xr2, xn2)
w = 3 * pi;
phi = 0;
%A = 5 * floor(t);
A = normpdf(t, t(round(end/2)), 3);

[xr2, xn2] = gen_sin(t, A, w, phi, r, snr);

% Select input data
global xr_train;
global xn_train;
global xr_test;
global xn_test;
global result_length;
global samples_div;
global predict_offset;
global snr_array;

xr_train = xr1;
xn_train = xn1;

xr_test = xr2;
xn_test = xn2;

result_length = 1;
predict_offset = 1;
samples_div = 1.5;

snr_array = [snr snr snr];

save_figure = false;

fprintf("Samples: %u Predict: %u Div: %.2f SNR size: %u\n", length(xr1), ...
    predict_offset, samples_div, length(snr_array));

% Plot input data:
if true
    fig_input = figure('Name', "Input data");
    tiledlayout(2, 1);

    nexttile;
    hold on;
    plot(t, xr_train, '-');
    plot(t, xn_train, '-x');
    legend("Train Data", "Train Measurements");
    hold off;

    nexttile;
    hold on;
    plot(t, xr_test, '-');
    plot(t, xn_test, '-x');
    legend("Test Data", "Test Measurements");
    hold off;
end

print_data_stats(xr_train, xn_train);
print_data_stats(xr_test, xn_test);

% Define variables to optimize:
%
% See also:
% https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html
minLayers = 2;
maxLayers = 3;

optimVars = [
    optimizableVariable('sequenceLength', [1 100], 'Type', 'integer', 'Optimize', true)

    optimizableVariable('numLayers', [minLayers maxLayers], 'Type', 'integer', 'Optimize', true)

    optimizableVariable('gradientThreshold', [0.001 10], 'Transform', 'log', 'Optimize', true)
    optimizableVariable('initialLearnRate', [0.0001 1], 'Transform', 'log', 'Optimize', true)
];

global optimize_type; optimize_type = false;
global optimize_type_default; optimize_type_default = "lstm";

global optimize_size; optimize_size = true;
global optimize_size_default; optimize_size_default = 10;

global optimize_act; optimize_act = true;
global optimize_act_default; optimize_act_default = "none";

global optimize_drop; optimize_drop = true;
global optimize_drop_default; optimize_drop_default = 0.00;

% Optimize layers structure cccording to maxLayers
for n = 1:maxLayers
    typeName = sprintf("ht%u", n);
    % "lstm" "gru"
    optimVars(end + 1,:) = optimizableVariable(typeName, ["lstm" "gru"], 'Optimize', optimize_type);

    hsName = sprintf("hs%u", n);
    optimVars(end + 1,:) = optimizableVariable(hsName, [1 100], 'Type', 'integer', 'Optimize', optimize_size);

    actName = sprintf("act%u", n);
    % "none" "relu" "tanh" "leakyrelu" "clippedrelu" "elu" "swish"
    optimVars(end + 1,:) = optimizableVariable(actName, ["none" "relu" "tanh"], 'Optimize', optimize_act);

    dropName = sprintf("drop%u", n);
    optimVars(end + 1,:) = optimizableVariable(dropName, [0 0.5], 'Optimize', optimize_drop);
end

disp(optimVars);

BayesObject = bayesopt(make_validation_fcn, optimVars,  ...
    'MaxObjectiveEvaluations', 30, ...
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
disp(net.Layers);
disp(opt);

sample_length = 1;

[test_samples, test_results] = prepare_train_data(xr_test, xn_test, sample_length, result_length, predict_offset);
X = test_network(net, test_samples, result_length, "lstm");

[xrs, X] = align_data(xr_test, X);

[error, abs_error, mse_array, rmse_array, max_error, mean_error, mse, rmse] = calc_errors(xrs, X);

fprintf("SL: %u HS: %u ME: %f MSE: %f\n", opt.sequenceLength, opt.hs1, mean_error, mse);
plot_results("Best NN configuration", t, xr_train, xr_test, xn_test, X, save_figure, samples_div);

function ObjFcn = make_validation_fcn()
ObjFcn = @validation_fcn;
    function [mean_error, cons, filename] = validation_fcn(optimVars)
        disp(optimVars);

        % Define input data
        sample_length = optimVars.sequenceLength;

        % Create neural network
        maxEpochs = 100;

        % Generate layers parameters
        ht_array = string.empty;
        hs_array = [];
        act_array = string.empty;
        drop_array = [];

        global optimize_type;
        global optimize_size;
        global optimize_act;
        global optimize_drop;

        global optimize_type_default;
        global optimize_size_default;
        global optimize_act_default;
        global optimize_drop_default;

        for n = 1:optimVars.numLayers
            if optimize_type
                ht_array(1, end + 1) = eval(sprintf("optimVars.ht%u", n));
            else
                ht_array(1, end + 1) = optimize_type_default;
            end

            if optimize_size
                hs_array(1, end + 1) = eval(sprintf("optimVars.hs%u", n));
            else
                hs_array(1, end + 1) = optimize_size_default;
            end

            if optimize_act
                act_array(1, end + 1) = eval(sprintf("optimVars.act%u", n));
            else
                act_array(1, end + 1) = optimize_act_default;
            end

            if optimize_drop
                drop_array(1, end + 1) = eval(sprintf("optimVars.drop%u", n));
            else
                drop_array(1, end + 1) = optimize_drop_default;
            end
        end

        layers = [ ...
            sequenceInputLayer(1)
            rnnBlock(hs_array, ht_array, act_array, drop_array, optimVars.numLayers)
            fullyConnectedLayer(1)
            regressionLayer
        ];
        disp(layers);

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

        % train_samples = samples; train_results = results;
        sample_length = 1;
        global xr_train;
        global xn_train;
        global xr_test;
        global xn_test;
        global result_length;
        global predict_offset;
        global samples_div;
        global snr_array;
        [samples, results] = prepare_train_data(xr_train, xn_train, sample_length, result_length, predict_offset, samples_div, snr_array);

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
        [test_samples, test_results] = prepare_train_data(xr_test, xn_test, sample_length, result_length, predict_offset);
        X = test_network(net, test_samples, result_length, "lstm");

        [xrs, X] = align_data(xr_test, X);

        [error, abs_error, mse_array, rmse_array, max_error, mean_error, mse, rmse] = calc_errors(xrs, X);

        fprintf("SL: %u HS: %u ME: %f MSE: %f\n", sample_length, hs_array, mean_error, mse);

        % Save trained network to file
        filename = "bayesopt/bayesopt_" + num2str(mean_error) + ".mat";
        fprintf("Saving to file: %s\n", filename);
        save(filename, 'net', 'mean_error', 'optimVars');

        cons = [];
    end
end
