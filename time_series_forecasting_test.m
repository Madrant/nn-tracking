clear all; % Clear variables
close all; % Close plots
clc;       % Clear command window

% Model options
start_time = 1;
time_step = 0.01;
end_time = 40;

% Initialize random number generator
rng(12345, 'combRecursive');

% Generate test data (real target position)
r = 0.01;
snr = 5;

t = start_time:time_step:end_time;

% Data set 1 (xr1, xr2)
w = 3 * pi;
phi = 0;
A = 0.5 + floor(t);

xr1 = A .* sin(w * t + phi);
xn1 = A .* (1 + normrnd(0, r)) .* sin(w * (1 + normrnd(0, r)) * t + phi*(1 + normrnd(0, r)));
xn1 = awgn(xn1, snr, 'measured');

[xr1, xn1] = rescale_data(xr1, xn1, 0, 1);

% Data set 2 (xr2, xn2)
w = 1 * pi;
phi = 0;
A = 1;

xr2 = A .* sin(w * t + phi);
xn2 = A .* (1 + normrnd(0, r)) .* sin(w * (1 + normrnd(0, r)) * t + phi*(1 + normrnd(0, r)));
xn2 = awgn(xn2, snr, 'measured');

[xr2, xn2] = rescale_data(xr2, xn2, 0, 1);

xlswrite("xr1.xlsx", xr1.');
xlswrite("xr2.xlsx", xr2.');

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

xr_train = xr1;
xn_train = xn1;

xr_test = xr1;
xn_test = xn1;

fprintf("Time: [%f:%f:%f]\n", start_time, time_step, end_time);
fprintf("SNR: %f\n", snr);

fprintf("Measurements Mean error: %f\n", mean(abs(xr - xn)));
fprintf("Measurements Max error:  %f\n", max(abs(xr - xn)));
fprintf("Measurements MSE:        %f\n", mean(xr - xn).^2);
fprintf("Measurements RMSE:       %f\n", sqrt(mean(xr - xn).^2));

% NN options
sample_length = 10;
result_length = 1;
samples_div = 1.5;

hiddenSize = 5;
maxEpochs = 100;

snr_array = [snr snr snr];

% Plot options
save_figure = 0;

% Enable/disable some various networks in test
en_nn_ff_ns = 0;
en_nn_ff_ts = 0;
en_nn_lstm_ns = 1;
en_nn_lstm_ts = 0;
en_nn_gru_ns = 0;
en_nn_gru_ts = 0;
en_nn_hybrid = 0;

for sample_length = [sample_length] %[1 3 5]
for hiddenSize = [hiddenSize] %[4, 5, 7, 10]

% Feedforward NN
if en_nn_ff_ns
    name = sprintf("FF NN - Noise Hs %u Samples %u Div %.2f", hiddenSize, sample_length, samples_div);

    nn_outputs = noise_ff_nn(t, xr_train, xn_train, xn_test, sample_length, result_length, samples_div, hiddenSize, maxEpochs, 'trainrp', snr_array);
    plot_results(name, t, xr_train, xr_test, xn_test, nn_outputs, save_figure);

    res_nn_ff_ns = nn_outputs;
end

if en_nn_ff_ts
    name = sprintf("FF NN - Prediction Hs %u Samples %u Div %.2f", hiddenSize, sample_length, samples_div);

    nn_outputs = ts_ff_nn(t, xr_train, xn_test, sample_length, result_length, samples_div, hiddenSize, maxEpochs, 'trainrp');
    plot_results(name, t, xr_train, xr_test, xn_test, nn_outputs, save_figure);

    res_nn_ff_ts = nn_outputs;
end

% LSTM NN
if en_nn_lstm_ns
    name = sprintf("LSTM NN - Noise Hs %u Samples %u Div %.2f", hiddenSize, sample_length, samples_div);

    nn_outputs = noise_lstm_nn(t, xr_train, xn_train, xn_test, sample_length, result_length, samples_div, hiddenSize, maxEpochs, "lstm", snr_array);
    plot_results(name, t, xr_train, xr_test, xn_test, nn_outputs, save_figure, 0, samples_div);

    res_nn_lstm_ns = nn_outputs;
end

if en_nn_lstm_ts
    name = sprintf("LSTM NN - Prediction Hs %u Samples %u Div %.2f", hiddenSize, sample_length, samples_div);

    nn_outputs = ts_lstm_nn(t, xr_train, xn_test, sample_length, result_length, samples_div, hiddenSize, maxEpochs, "lstm");
    plot_results(name, t, xr_train, xr_test, xn_test, nn_outputs, save_figure);

    res_nn_lstm_ts = nn_outputs;
end

% GRU NN
if en_nn_gru_ns
    name = sprintf("GRU NN - Noise Hs %u Samples %u Div %.2f", hiddenSize, sample_length, samples_div);
    nn_outputs = noise_lstm_nn(t, xr_train, xn_train, xn_test, sample_length, result_length, samples_div, hiddenSize, maxEpochs, "gru", snr_array);
    plot_results(name, t, xr_train, xr_test, xn_test, nn_outputs, save_figure);

    res_nn_gru_ns = nn_outputs;
end

if en_nn_gru_ts
    name = sprintf("GRU NN - Prediction Hs %u Samples %u Div %.2f", hiddenSize, sample_length, samples_div);
    nn_outputs = ts_lstm_nn(t, xr_train, xn_train, sample_length, result_length, samples_div, hiddenSize, maxEpochs, "gru");
    plot_results(name, t, xr_train, xr_test, xn_test, nn_outputs, save_figure);
    
    res_nn_gru_ts = nn_outputs;
end

if en_nn_hybrid
    sl_array = [sample_length 5];
    rl_array = [result_length 1];
    hs_array = [hiddenSize 11];
    snr_array = [snr snr snr];
    % samples_div = 1.5;

    name = sprintf("Hybrid LSTM NN - Noise Hs %u Samples %u Div %.2f", hiddenSize, sample_length, samples_div);

    [net1_outputs, net2_outputs] = hybrid_lstm_nn(...
        t, xr_train, xn_train, xn_test, ...
        sl_array, rl_array, samples_div, ...
        hs_array, maxEpochs, "lstm", snr_array);

    %plot_results(name, t, xr_train, xr_test, xn_test, net1_outputs, save_figure, 0);
    plot_results(name, t, xr_train, xr_test, xn_test, net2_outputs, save_figure, 0, samples_div);
end

end % hiddenSize
end % sample_length

% Kalman filter
% kf_outputs = ts_kf(t, xr_train, xn_test);
% plot_results("KF", t, xr_train, xr_test, xn_test, kf_outputs, save_figure, 0);

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
    plot(t, xr_test, '-');
    plot(t, xn_test, '-x');

    if en_nn_ff_ts,   plot(t(sample_length + 1:length(t)), res_nn_ff_ts), end
    if en_nn_lstm_ts, plot(t(sample_length + 1:length(t)), res_nn_lstm_ts), end
    if en_nn_gru_ts,  plot(t(sample_length + 1:length(t)), res_nn_gru_ts), end

    legend("Data", "Measurements", "FF NN", "LSTM", "GRU");
    hold off;
end