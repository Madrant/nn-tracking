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

% Initialize random number generator
% rng(12345, 'combRecursive');

% Some (not all) training functions:
% 'trainlm'	Levenberg-Marquardt
% 'trainrp'	Resilient Backpropagation
% 'traingd'	Gradient Descent
trainFcn = 'trainrp';
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

fprintf("Time: [%f:%f:%f]\n", start_time, time_step, end_time);
fprintf("SNR: %f\n", snr);

% Calculate Mean Max Error, MSE, RMSE for hiddenSizes
hs = 10;

mean_me_array = zeros(length(hs), 1);
mean_mse_array = zeros(length(hs), 1);
mean_rmse_array = zeros(length(hs), 1);
mean_epochs_array = zeros(length(hs), 1);

for n = 1:length(hs)
    loops = 10;

    me_array = zeros(loops, 1);
    mse_array = zeros(loops, 1);
    rmse_array = zeros(loops, 1);
    epochs_array = zeros(loops, 1);

    for l = 1:loops
        [max_error, mse, rmse, tr] = time_series_forecasting(t, x, xn, sample_length, result_length, samples_div, hs(n), trainFcn);

        me_array(l) = max_error;
        mse_array(l) = mse;
        rmse_array(l) = rmse;
        epochs_array(l) = tr.num_epochs;
    end

    mean_me_array(n) = mean(me_array);
    mean_mse_array(n) = mean(mse_array);
    mean_rmse_array(n) = mean(rmse_array);
    mean_epochs_array(n) = mean(epochs_array);

    %waitforbuttonpress;
end

% Plot Error and Epochs dependencies from hiddenSize
hs_mean_error = figure('name', ['Error and Epochs dependencies from hiddenSize for trainFcn: ', trainFcn]);
tiledlayout(2, 1);

nexttile;
hold on;
plot(hs, mean_me_array, '-x');
plot(hs, mean_mse_array, '-x');
plot(hs, mean_rmse_array, '-x');

xlabel('Hidden layer size');
ylabel('Error');
legend('Max Error', 'MSE', 'RMSE');
hold off;

nexttile;
plot(hs, mean_epochs_array, '-x');

xlabel('Training epochs');
ylabel('Error');
legend('Train epochs');

% Save figure to file
date_str = datestr(datetime(), 'yyyymmdd_HHMMSS');
str_name = sprintf('hs_mean_error_func_%s_hs_%u_%u_loops_%u_date_%s', trainFcn, hs(1), hs(end), loops, date_str);
saveas(hs_mean_error, str_name + ".png");
