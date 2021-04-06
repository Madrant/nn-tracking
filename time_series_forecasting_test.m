clear all; % Clear variables
close all; % Close plots
clc;       % Clear command window

% Model options
start_time = 0;
time_step = 0.1;
end_time = 10;

% Initialize random number generator
rng(12345, 'combRecursive');

% Generate training data (filter model)
t = start_time:time_step:end_time;

w = 1 * pi;
phi = 0;
A = floor(t);
xt = A .* sin(w * t + phi);

% Generate test data (real target position)
w = 1 * pi;
phi = 0;
A = floor(t);

r = 0.01;
snr = 20;

xr1 = A .* sin(w * t + phi);
xn1 = A .* (1 + normrnd(0, r)) .* sin(w * (1 + normrnd(0, r)) * t + phi*(1 + normrnd(0, r)));
xn1 = awgn(xn1, snr, 'measured');

w = 2 * pi;
phi = 0;
A = 2;
xr2 = A .* sin(w * t + phi);

xn2 = A .* (1 + normrnd(0, r)) .* sin(w * (1 + normrnd(0, r)) * t + phi*(1 + normrnd(0, r)));
xn2 = awgn(xn2, snr, 'measured');

% Plot input data:
fig_input = figure('Name', "Input data");
hold on;
plot(t, xr1, '-d');
plot(t, xn1, '-x');
plot(t, xr2, '--');
plot(t, xn2, '--*');
legend("Data 1", "Measurements 1", "Data 2", "Measurements 2");
hold off;

% Select input data
xr = xr1;
xn = xn1;

fprintf("Time: [%f:%f:%f]\n", start_time, time_step, end_time);
fprintf("SNR: %f\n", snr);

fprintf("Measurements Max error: %f\n", max(abs(xr - xn)));
fprintf("Measurements MSE:       %f\n", mean(xr - xn).^2);
fprintf("Measurements RMSE:      %f\n", sqrt(mean(xr - xn).^2));

% NN options
sample_length = 3;
result_length = 1;
samples_div = 1;

% Plot options
save_figure = 0;

% Feedforward NN
nn_outputs = ts_ff_nn(t, xt, xn, sample_length, result_length, samples_div, 4, 'trainrp');
plot_results("FF NN 3", t, xt, xr, xn, nn_outputs, save_figure, sample_length);

% LSTM NN
%outputs = ts_lstm_nn(t, xt, xn, sample_length, result_length, samples_div);
%plot_results("LSTM NN", t, xt, xr, xn, outputs, save_figure, sample_length);

% Kalman filter
kf_outputs = ts_kf(t, xt, xn);
plot_results("KF", t, xt, xr, xn, kf_outputs, save_figure, 0);

% Extrapolation
outputs = ts_extrap(t, xr, xn, 'linear', 3);
plot_results("Extrapolation: Linear Points: 3", t, xt, xr, xn, outputs, save_figure, 3);

outputs = ts_extrap(t, xr, xn, 'spline', 3);
plot_results("Extrapolation: Spline Points: 3", t, xt, xr, xn, outputs, save_figure, 3);

outputs = ts_extrap(t, xr, xn, 'spline', 5);
plot_results("Extrapolation: Spline Points: 5", t, xt, xr, xn, outputs, save_figure, 5);

% Compare filter outputs

% Plot input data:
fig_outputs = figure('Name', "Filter outputs");
hold on;
plot(t, xr, '-');
plot(t, xn, '-x');
plot(t(sample_length + 1:length(t)), nn_outputs, '-o');
plot(t, kf_outputs, '-*');
legend("Data", "Measurements", "FF NN 3", "KF");
hold off;
