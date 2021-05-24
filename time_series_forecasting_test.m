clear all; % Clear variables
close all; % Close plots
clc;       % Clear command window

% Model options
start_time = 1;
time_step = 0.1;
end_time = 10;

% Initialize random number generator
rng(12345, 'combRecursive');

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

% Plot input data:
if true
    fig_input = figure('Name', "Input data");
    tiledlayout(2, 1);

    nexttile;
    hold on;
    grid on;
    plot(t, xr1, '-d');
    plot(t, xn1, '-x');
    legend("Train Data", "Train Measurements");
    hold off;

    nexttile;
    hold on;
    plot(t, xr2, '-');
    plot(t, xn2, '-x');
    legend("Test Data", "Test Measurements");
    hold off;
end

% Select input data
xr = xr1;
xn = xn1;

xr_train = xr1;
xn_train = xn1;

xr_test = xr2;
xn_test = xn2;

fprintf("Time: [%f:%f:%f] SNR: %f\n", start_time, time_step, end_time, snr);
print_data_stats(xr, xn);

speed_array = zeros(1, length(xr));

for n = 2:length(xr)
    dx = xr(n) - xr(n - 1);
    dt = t(n) - t (n - 1);

    speed = dx/dt;
    speed_array(1,n) = abs(speed);
end

% NN options
sample_length = 7;
result_length = 1;
samples_div = 1.5;
predict_offset = 1;

hiddenSize = 10;
maxEpochs = 100;

snr_array = [snr snr snr];

% Plot options
save_figure = 0;

% Enable/disable some various networks in test
en_nn_ff_ns = 1;
en_nn_lstm_ns = 1;
en_nn_gru_ns = 1;
en_nn_lstm_dl = 0;

for sample_length = [sample_length] %[1 3 5]
for hiddenSize = [hiddenSize] %[4, 5, 7, 10]

% Feedforward
if en_nn_ff_ns
    name = sprintf("FF NN - Noise Hs %u Samples %u Div %.2f", hiddenSize, sample_length, samples_div);

    nn_outputs = noise_ff_nn(t, xr_train, xn_train, xn_test, sample_length, result_length, predict_offset, samples_div, hiddenSize, maxEpochs, 'trainrp', snr_array);
    plot_results(name, t, xr_train, xr_test, xn_test, nn_outputs, save_figure, samples_div);

    res_nn_ff_ns = nn_outputs;
end

% LSTM
if en_nn_lstm_ns
    name = sprintf("LSTM NN - Noise Hs %u Samples %u Div %.2f", hiddenSize, sample_length, samples_div);

    [nn_outputs, train_samples] = noise_lstm_nn(t, xr_train, xn_train, xn_test, sample_length, result_length, predict_offset, samples_div, hiddenSize, maxEpochs, "lstm", snr_array);
    plot_results(name, t, xr_train, xr_test, xn_test, nn_outputs, save_figure, samples_div);

    res_nn_lstm_ns = nn_outputs;
end

% Deep LSTM
if en_nn_lstm_dl
    name = sprintf("LSTM NN - Deep Hs %u Samples %u Div %.2f", hiddenSize, sample_length, samples_div);

    [nn_outputs, train_samples] = deep_lstm_nn(t, xr_train, xn_train, xn_test, sample_length, result_length, predict_offset, samples_div, hiddenSize, maxEpochs, "lstm", snr_array);
    plot_results(name, t, xr_train, xr_test, xn_test, nn_outputs, save_figure, samples_div);

    res_nn_lstm_dl = nn_outputs;
end

% GRU
if en_nn_gru_ns
    name = sprintf("GRU NN - Noise Hs %u Samples %u Div %.2f", hiddenSize, sample_length, samples_div);
    nn_outputs = noise_lstm_nn(t, xr_train, xn_train, xn_test, sample_length, result_length, predict_offset, samples_div, hiddenSize, maxEpochs, "gru", snr_array);
    plot_results(name, t, xr_train, xr_test, xn_test, nn_outputs, save_figure, samples_div);

    res_nn_gru_ns = nn_outputs;
end

end % hiddenSize
end % sample_length

% Kalman filter
kf_outputs = ts_kf(t, xr_train, xn_test, predict_offset);
plot_results("KF", t, xr_train, xr_test, xn_test, kf_outputs, save_figure);

% Extrapolate KF output
if false && predict_offset > 0
    method = 'spline'; % linear, nearest, previous, pchip, cubic, v5cubic, makima, spline
    points = 5;

    name = sprintf("KF Extrapolation: Method: %s Points: %u", method, points);

    ex_kf_outputs = ts_kf(t, xr_train, xn_test);
    extrap_kf_outputs = ts_extrap(t, ex_kf_outputs, method, points, predict_offset);
    plot_results(name, t, xr_train, xr_test, xn_test, extrap_kf_outputs, save_figure, 1, 3);
end

% Compare filter outputs

% Plot input data:
if true
    fig_outputs = figure('Name', "Filter outputs");
    hold on;
    plot(t, xr_test, '-');
    plot(t, xn_test, '-x');

    plot(t, kf_outputs, '-d');

    if en_nn_ff_ns, plot(t(predict_offset + sample_length:length(t)), res_nn_ff_ns, '-*'), end
    if en_nn_lstm_ns, plot(t(predict_offset + 1:length(t)), res_nn_lstm_ns, '-*'), end
    if en_nn_gru_ns, plot(t(predict_offset + 1:length(t)), res_nn_gru_ns, '-+'), end
    if en_nn_lstm_dl, plot(t(predict_offset + 1:length(t)), res_nn_lstm_dl, '-o'), end

    legend("Data", "Measurements", "KF", "FF", "LSTM", "GRU");
    hold off;
end