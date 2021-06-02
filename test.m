%clear all;
close all;
clc;

% Model options
start_time = 1;
time_step = 0.1;
end_time = 10;

% Initialize random number generator
%rng(12345, 'combRecursive');

% Generate test data (real target position)
r = 0.01;
snr = 10;

t = start_time:time_step:end_time;

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
xr = xr1;
xn = xn1;

xr_train = xr1;
xn_train = xn1;

xr_test = xr1;
xn_test = xn1;

if false
    plot_input_data(t, xr_train, xn_train, xr_test, xn_test);
end

train_data = struct('t', num2cell(t), 'xn', num2cell(xn), 'xr', num2cell(xr));
test_data = struct('t', num2cell(t), 'xn', num2cell(xn), 'xr', num2cell(xr));

% https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html
options = trainingOptions('adam', ... % sgdm, rmsprop, adam
    'MaxEpochs', 100, ...
    'SequenceLength', 30, ...
    'GradientThreshold', 1, ...
    'Verbose', 0, ...
    'Plots', 'none', ... % 'training-progress', 'none'
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Shuffle', 'once', ...
    'ExecutionEnvironment', 'cpu');

layers = [ ...
        sequenceInputLayer(2)
        lstmLayer(10)
        fullyConnectedLayer(1)
        regressionLayer
    ];

predict_offset = 1;
samples_div = 1.5;
loss_prob = 0.00;
snr = 5;

train_data = prepare_train_data(train_data, predict_offset, samples_div, 5, 5, 0, loss_prob, snr);
test_data = prepare_train_data(test_data, predict_offset, 1, 5, 5, 0, loss_prob, snr);

train_input = struct_fields_to_cell_array(train_data, ["dt" "xn"]).';
train_output = struct_fields_to_cell_array(train_data, ["xr"]).';

test_input = struct_fields_to_cell_array(test_data, ["dt" "xn"]).';
test_output = struct_fields_to_cell_array(test_data, ["xr"]).';

fprintf("Train start");
net = trainNetwork(train_input, train_output, layers, options);
fprintf("Train end");

% Get network output
num_outputs = 1;
net_outputs = test_network(net, test_input, num_outputs, "lstm");

% Plot network output in comparison with inputs
plot_results("Test", t, xr_train, xr_test, xn_test, net_outputs, false, samples_div);
