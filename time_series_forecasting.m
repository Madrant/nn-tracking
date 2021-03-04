clear all; % Clear variables
close all; % Close plots
clc;       % Clear command window

% Model options
sample_length = 5;
result_length = 1;
samples_div = 1;
test_step = 1;
snr = 5;

start_time = 0;
end_time = 10;
time_step = 0.1;

rnd_seed = 12345;

% Some (not all) training functions:
% 'trainlm'	Levenberg-Marquardt
% 'trainrp'	Resilient Backpropagation
% 'traingd'	Gradient Descent
trainFcn = 'trainlm';
hiddenSizes = 10;

% Print options
fprintf("Time: [%f:%f:%f]\n", start_time, time_step, end_time);
fprintf("Samples: [%f:%f] Train sample div: %f\n", sample_length, result_length, samples_div);
fprintf("SNR: %f\n", snr);
fprintf("Network: Hidden: %f Train: '%s'\n", hiddenSizes, trainFcn);

% Generate training data (real target position)
t = start_time:time_step:end_time;

w = 1 * pi;
phi = 0;
A = floor(t);
x = A.*sin(w * t + phi);

% Generate test data ( noised measurements)
tn = t;
xn = awgn(x, snr);

fprintf("Measurements Max error: %f\n", max(abs(x - xn)));
fprintf("Measurements MSE:       %f\n", mean(x - xn).^2);
fprintf("Measurements RMSE:      %f\n", sqrt(mean(x - xn).^2));

% Setup plot layout
tiledlayout(3, 1);

% Plot input data
nexttile;
hold on;
plot(t, x);
plot(t, xn, 'x'); % Plot measurements
title('Measurements');
legend('Training sample', 'Test sample');
hold off;

%waitforbuttonpress;

% Initial weights are random unless we initialize random number generator
rng(rnd_seed, 'combRecursive');

% Create neural network
net = feedforwardnet(hiddenSizes, trainFcn);

% Prepare train data set
samples_num = round(length(x) / samples_div) - (sample_length + result_length - 1);

samples = zeros(samples_num, sample_length);
results = zeros(samples_num, result_length);

for n = 1 : samples_num
    samples(n,:) = x(n:n + sample_length - 1);
    results(n,:) = x(n + sample_length:n + sample_length + result_length - 1);
end

% Plot training sample length
% plot(t(1:n), x(1:n));

% Print generated test data
% fprintf("samples: "); disp(samples);
% fprintf("results: "); disp(results);

% Transpose test arrays to fit network inputs
samples = samples.';
results = results.';

% Configure network inputs:
net = configure(net, samples, results);
fprintf("net.inputs: %d\n", net.inputs{1}.size);

% Train network
net = train(net, samples, results);

% Test network
samples_num = round(length(x) / test_step) - (sample_length + result_length - 1);

measurements = zeros(samples_num, result_length);
net_outputs = zeros(samples_num, result_length);

for n = 1 : test_step: length(xn) - (sample_length + result_length - 1)
    test_sample = xn(n:n + sample_length - 1);
    measurement = xn(n + sample_length:n + sample_length + result_length - 1);

    net_output = net(test_sample.').';

    measurements(n,:) = measurement;
    net_outputs(n,:) = net_output;

    % fprintf("test_sample: "); disp(test_sample);
    % fprintf("net_output: "); disp(net_output);
    % fprintf("measurement: "); disp(measurement);

    % Plot network result
    % sample_time = t(n):time_step:t(n + length(test_sample) - 1);
    % result_time = t(n + sample_length):time_step:t(n + sample_length + length(net_output) - 1);

    % plot(sample_time, test_sample, 'x'); waitforbuttonpress;
    % plot(result_time, net_output, 'o'); waitforbuttonpress;
end

% Plot network output
nexttile;
hold on;
plot(t(sample_length + 1:length(t)), measurements);
plot(t(sample_length + 1:length(t)), net_outputs, 'o');
title('Network output');
legend('Test sample', 'Network output');
hold off;

% Assess the performance of the trained network.
%
% The default performance function is mean squared error.
perf = perform(net, measurements, net_outputs);

% Calculate absolute errors
error = (measurements - net_outputs);
abs_error = abs(error);
max_error = max(abs_error);
mse = mean(error.^2);
rmse = sqrt(mse);

fprintf("Network performance (MSE by defaults): "); disp(perf);
fprintf("Max error: %f\n", max_error);
fprintf("MSE:       %f\n", mse);
fprintf("RMSE:      %f\n", rmse);

% Plot absolute error
nexttile;
plot(t(sample_length + 1:length(t)), abs_error);
title('Absolute error');
