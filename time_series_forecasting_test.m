clear all; % Clear variables
close all; % Close plots
clc;       % Clear command window

% Model options
sample_length = 5;
result_length = 1;
samples_div = 1;
snr = 10;

start_time = 0;
end_time = 10;
time_step = 0.1;

rnd_seed = 12345;

% Some (not all) training functions:
% 'trainlm'	Levenberg-Marquardt
% 'trainrp'	Resilient Backpropagation
% 'traingd'	Gradient Descent
trainFcn = 'traingd';
hiddenSizes = 10;

% Generate training data (real target position)
t = start_time:time_step:end_time;

w = 1 * pi;
phi = 0;
A = floor(t);
x = A.*sin(w * t + phi);

% Generate test data ( noised measurements)
tn = t;
xn = awgn(x, snr);

% Initialize random number generator
rng(rnd_seed, 'combRecursive');

fprintf("Time: [%f:%f:%f]\n", start_time, time_step, end_time);
fprintf("SNR: %f\n", snr);

[max_error, mse] = time_series_forecasting(t, x, xn, sample_length, result_length, samples_div, hiddenSizes, trainFcn)
