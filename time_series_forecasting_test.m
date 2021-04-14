clear all; % Clear variables
close all; % Close plots
clc;       % Clear command window

% Model options
start_time = 0;
time_step = 0.01;
end_time = 10;

% Initialize random number generator
rng(12345, 'combRecursive');

% Generate training data (filter model)
t = start_time:time_step:end_time;

w = 2 * pi;
phi = 0;
A = floor(t);

xt = A .* sin(w * t + phi);
xt = normalize(xt, 'range');

% Generate test data (real target position)
r = 0.01;
snr = 5;

% Data set 1 (xr1, xr2)
w = 2 * pi;
phi = 0;
A = floor(t);

xr1 = A .* sin(w * t + phi);
xn1 = A .* (1 + normrnd(0, r)) .* sin(w * (1 + normrnd(0, r)) * t + phi*(1 + normrnd(0, r)));
xn1 = awgn(xn1, snr, 'measured');

rescale_array = rescale([xr1; xn1], 0, 1);
xr1 = rescale_array(1,:);
xn1 = rescale_array(2,:);

% Data set 2 (xr2, xn2)
w = 1 * pi;
phi = 0;
A = 1;

xr2 = A .* sin(w * t + phi);
xn2 = A .* (1 + normrnd(0, r)) .* sin(w * (1 + normrnd(0, r)) * t + phi*(1 + normrnd(0, r)));
xn2 = awgn(xn2, snr, 'measured');

rescale_array = rescale([xr2; xn2], 0, 1);
xr2 = rescale_array(1,:);
xn2 = rescale_array(2,:);

% Plot input data:
fig_input = figure('Name', "Input data");
tiledlayout(2, 1);

nexttile;
hold on;
plot(t, xr1, '-');
plot(t, xn1, '-x');
legend("Data 1", "Measurements 1");
hold off;

nexttile;
hold on;
plot(t, xr2, '-');
plot(t, xn2, '-x');
legend("Data 2", "Measurements 2");
hold off;

% Select input data
xr = xr1;
xn = xn1;

fprintf("Time: [%f:%f:%f]\n", start_time, time_step, end_time);
fprintf("SNR: %f\n", snr);

fprintf("Measurements Max error:  %f\n", max(abs(xr - xn)));
fprintf("Measurements Mean error: %f\n", mean(abs(xr - xn)));
fprintf("Measurements MSE:        %f\n", mean(xr - xn).^2);
fprintf("Measurements RMSE:       %f\n", sqrt(mean(xr - xn).^2));

% NN options
sample_length = 5;
result_length = 1;
samples_div = 1.5;

hiddenSize = 7;
maxEpochs = 100;

% Plot options
save_figure = 0;

% Enable/disable some various networks in test
en_nn_ff_ns = 0;
en_nn_ff_ts = 0;
en_nn_lstm_ns = 0;
en_nn_lstm_ts = 0;
en_nn_gru_ns = 1;
en_nn_gru_ts = 0;

for sample_length = [sample_length] %[1 3 5]
for hiddenSize = [hiddenSize] %[4, 5, 7, 10]

% Feedforward NN
if en_nn_ff_ns
    name = sprintf("FF NN - Noise Hs %u Samples %u Div %.2f", hiddenSize, sample_length, samples_div);
    un_outputs = noise_ff_nn(t, xr, xn, sample_length, result_length, samples_div, hiddenSize, maxEpochs, 'trainrp');
    plot_results(name, t, xt, xr, xn, un_outputs, save_figure, sample_length);

    res_nn_ff_ns = un_outputs;
end

if en_nn_ff_ts
    name = sprintf("FF NN - Prediction Hs %u Samples %u Div %.2f", hiddenSize, sample_length, samples_div);
    nn_outputs = ts_ff_nn(t, xt, xn, sample_length, result_length, samples_div, hiddenSize, maxEpochs, 'trainrp');
    plot_results(name, t, xt, xr, xn, nn_outputs, save_figure, sample_length);

    res_nn_ff_ts = nn_outputs;
end

% LSTM NN
if en_nn_lstm_ns
    name = sprintf("LSTM NN - Noise Hs %u Samples %u Div %.2f", hiddenSize, sample_length, samples_div);
    un_outputs = noise_lstm_nn(t, xr, xn, xn, sample_length, result_length, samples_div, hiddenSize, maxEpochs, "lstm");
    plot_results(name, t, xt, xr, xn, un_outputs, save_figure, sample_length);

    res_nn_lstm_ns = un_outputs;
end

if en_nn_lstm_ts
    name = sprintf("LSTM NN - Prediction Hs %u Samples %u Div %.2f", hiddenSize, sample_length, samples_div);
    nn_outputs = ts_lstm_nn(t, xt, xn, sample_length, result_length, samples_div, hiddenSize, maxEpochs, "lstm");
    plot_results(name, t, xt, xr, xn, nn_outputs, save_figure, sample_length);

    res_nn_lstm_ts = nn_outputs;
end

% GRU NN
if en_nn_gru_ns
    name = sprintf("GRU NN - Noise Hs %u Samples %u Div %.2f", hiddenSize, sample_length, samples_div);
    un_outputs = noise_lstm_nn(t, xr, xn, xn, sample_length, result_length, samples_div, hiddenSize, maxEpochs, "gru");
    plot_results(name, t, xt, xr, xn, un_outputs, save_figure, sample_length);

    res_nn_gru_ns = un_outputs;
end

if en_nn_gru_ts
    name = sprintf("GRU NN - Prediction Hs %u Samples %u Div %.2f", hiddenSize, sample_length, samples_div);
    nn_outputs = ts_lstm_nn(t, xt, xn, sample_length, result_length, samples_div, hiddenSize, maxEpochs, "gru");
    plot_results(name, t, xt, xr, xn, nn_outputs, save_figure, sample_length);
    
    res_nn_gru_ts = nn_outputs;
end

end % hiddenSize
end % sample_length

% Kalman filter
kf_outputs = ts_kf(t, xt, xn);
plot_results("KF", t, xt, xr, xn, kf_outputs, save_figure, 0);

% Extrapolation
% outputs = ts_extrap(t, xr, xn, 'linear', 3);
% plot_results("Extrapolation: Linear Points: 3", t, xt, xr, xn, outputs, save_figure, 3);

% outputs = ts_extrap(t, xr, xn, 'spline', 3);
% plot_results("Extrapolation: Spline Points: 3", t, xt, xr, xn, outputs, save_figure, 3);

% outputs = ts_extrap(t, xr, xn, 'spline', 5);
% plot_results("Extrapolation: Spline Points: 5", t, xt, xr, xn, outputs, save_figure, 5);

% Compare filter outputs

% Plot input data:
if false
    fig_outputs = figure('Name', "Filter outputs");
    hold on;
    plot(t, xr, '-');
    plot(t, xn, '-x');

    if en_nn_ff_ts,   plot(t(sample_length + 1:length(t)), res_nn_ff_ts), end
    if en_nn_lstm_ts, plot(t(sample_length + 1:length(t)), res_nn_lstm_ts), end
    if en_nn_gru_ts,  plot(t(sample_length + 1:length(t)), res_nn_gru_ts), end

    legend("Data", "Measurements", "FF NN", "LSTM", "GRU");
    hold off;
end