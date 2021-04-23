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
result_length = 1;
samples_div = 1.5;
maxEpochs = 100;

snr_array = [snr snr snr];

perf_data = [];

for sl = [2 3 4 5 7 9 15]
for hs = 5:5:50
    % Initialize random number generator for each step
    % to get reproducable results
    rng(12345, 'combRecursive');

    [X, train_samples] = deep_lstm_nn(...
        t, xr_train, xn_train, xn_test, ...
        sl, result_length, samples_div, ...
        hs, maxEpochs, "lstm", snr_array ...
    );

    [ts, X] = align_data(t, X);
    [xrs, X]= align_data(xr, X);

    [error, abs_error, mse_array, rmse_array, max_error, mean_error, mse, rmse] = calc_errors(xrs, X);

    fprintf("SL: %u HS: %u ME: %f MSE: %f\n", sl, hs, mean_error, mse);

    perf_data(end + 1,:)  = [sl hs mean_error mse];
end % hiddenSize
end % sample_length

% Check for minimal mean error and MSE
[m, i] = min(perf_data(:,3));
fprintf("Min ME:  %f \tSL: %u \tHS: %u\n", m, perf_data(i,1), perf_data(i,2));

[m, i] = min(perf_data(:,4));
fprintf("Min MSE: %f \tSL: %u \tHS: %u\n", m, perf_data(i,1), perf_data(i,2));

% Sort collected data by MSE
perf_data_sorted = sortrows(perf_data, 4);
disp(perf_data_sorted);

me_data = perf_data;
mse_data = perf_data; mse_data(:,3) = mse_data(:,4);

plot_2var_dep(me_data, ["sl" "hs" "ME"]);
plot_2var_dep(mse_data, ["sl" "hs" "MSE"]);
