clear all; % Clear variables
close all; % Close plots
clc;       % Clear command window

% Model options
sample_length = 5;
result_length = 1;
samples_div = 1;
snr = 20;

start_time = 0;
end_time = 40;
time_step = 0.1;

% Initialize random number generator
rng(12345, 'combRecursive');

% Generate training data (real target position)
t = start_time:time_step:end_time;

w = 1 * pi;
phi = 0;
A = floor(t);
x = A .* sin(w * t + phi);

% Generate test data ( noised measurements)
tn = t;
r = 0.01;
xn = A .* (1 + normrnd(0, r)) .* sin(w * (1 + normrnd(0, r)) * t + phi*(1 + normrnd(0, r)));
xn = awgn(xn, snr, 'measured');

fprintf("Time: [%f:%f:%f]\n", start_time, time_step, end_time);
fprintf("SNR: %f\n", snr);

% Calculate Mean Max Error, MSE, RMSE for various hiddenSizes 
% and train functions
hs = 10;
tf = ["trainlm"];
loops = 1;
save_figure = 0;

outputs = time_series_forecasting(t, x, xn, sample_length, result_length, samples_div, hs, tf);

% t(sample_length + 1:length(t))
plot_results("FF NN", t, x, xn, outputs, save_figure, sample_length);


