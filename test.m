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

xr_test = xr2;
xn_test = xn2;

if false
    plot_input_data(t, xr_train, xn_train, xr_test, xn_test);
end

data = struct('t', num2cell(t), 'xr', num2cell(xr), 'xn', num2cell(xn));

% https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html
options = trainingOptions('adam', ... % sgdm, rmsprop, adam
    'MaxEpochs', 50, ...
    'SequenceLength', 10, ...
    'GradientThreshold', 1, ...
    'Verbose', 0, ...
    'Plots', 'none', ... % 'training-progress', 'none'
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Shuffle', 'once');

layers = [ ...
        sequenceInputLayer(2)
        lstmLayer(10)
        fullyConnectedLayer(1)
        regressionLayer
    ];

c_train = struct_fields_to_cell_array(data, ["t" "xn"]).';
c_test = struct_fields_to_cell_array(data, ["xr"]).';

fprintf("Train start");
%net = trainNetwork(c_train, c_test, layers, options);
fprintf("Train end");

% Get network output
net_outputs = test_network(net, c_train, 1, "lstm");
disp(net_outputs);

% Plot network output in comparison with inputs
plot_results("Test", t, xr_train, xr_test, xn_test, net_outputs);
