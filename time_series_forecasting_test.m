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
trainFcn = 'trainlm';
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
hs = 1:2:30;

mean_me_array = zeros(length(hs), 1);
mean_mse_array = zeros(length(hs), 1);
mean_rmse_array = zeros(length(hs), 1);

for n = 1:length(hs)
    loops = 10;

    me_array = zeros(loops, 1);
    mse_array = zeros(loops, 1);
    rmse_array = zeros(loops, 1);

    for l = 1:loops
        [max_error, mse, rmse] = time_series_forecasting(t, x, xn, sample_length, result_length, samples_div, hs(n), trainFcn);

        me_array(l) = max_error;
        mse_array(l) = mse;
        rmse_array(l) = rmse;
    end

    mean_me_array(n) = mean(me_array);
    mean_mse_array(n) = mean(mse_array);
    mean_rmse_array(n) = mean(rmse_array);

    %waitforbuttonpress;
end

hs_mean_error = figure('name', ['Error dependencies from hiddenSize for trainFcn: ', trainFcn]);
hold on;
plot(hs, mean_me_array, '-x');
plot(hs, mean_mse_array, '-x');
plot(hs, mean_rmse_array, '-x');
legend('Max Error', 'MSE', 'RMSE');
hold off;

% Save figure to file
date_str = datestr(datetime(), 'yyyymmdd_HHMMSS');
str_name = sprintf('hs_mean_error_func_%s_hs_%u_%u_loops_%u_date_%s', trainFcn, hs(1), hs(end), loops, date_str);
saveas(hs_mean_error, str_name + ".png");
