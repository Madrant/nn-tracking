clear all; % Clear variables
close all; % Close plots
clc;       % Clear command window

% Initialize random number generator
rng(12345, 'combRecursive');

% Model time
start_time = 1;
time_step = 0.1;
end_time = 10;

t = start_time:time_step:end_time;

% Generate test data (real target position)
r = 0.01;
snr = 5;

% Data set 1 (xr1, xr2)
w = 3 * pi;
phi = 0;
A = 0.5 + floor(t);

[xr1, xn1] = gen_sin(t, A, w, phi, r, snr);

% Select input data
xr = xr1;
xn = xn1;

xr_train = xr1;
xn_train = xn1;

xr_test = xr1;
xn_test = xn1;

fprintf("Time: [%f:%f:%f] SNR: %f\n", start_time, time_step, end_time, snr);
print_data_stats(xr, xn);

% NN options
sample_length = 5;
result_length = 1;
samples_div = 1.5;

hiddenSize = 5;
maxEpochs = 100;

snr_array = [snr snr snr];

for sl = 1:30 %[1 3 5]
for hs = 1:30 %[4, 5, 7, 10]
    [X, train_samples] = deep_lstm_nn(...
        t, xr_train, xn_train, xn_test, ...
        sl, result_length, samples_div, ...
        hs, maxEpochs, "lstm", snr_array ...
    );

    [ts, X] = align_data(t, X);
    [xrs, X]= align_data(xr, X);

    [error, abs_error, mse_array, rmse_array] = calc_errors(xrs, X);

    % Calculate error values
    max_error = max(abs_error);
    mean_error = mean(abs_error);
    mse = mean(error.^2);
    rmse = sqrt(mse);

    fprintf("SL: %u HS: %u ME: %f MSE: %f\n", sl, hs, mean_error, mse);
end % hiddenSize
end % sample_length
