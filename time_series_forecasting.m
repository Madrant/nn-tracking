clear all; % Clear variables
close all; % Close plots
clc;       % Clear command window

% Model options
sample_length = 5;
result_length = 1;
samples_div = 2;
test_step = 10;

start_time = 0;
end_time = 10;
time_step = 0.01;

% Generate train data
t = start_time:time_step:end_time; % Time

w = 1 * pi;
phi = 0;
A = floor(t);

x = A.*sin(w * t + phi);

% Generate test data
tn = t;
xn = awgn(x, 30);

% Plot input data
hold on;
plot(t, x);
plot(t, xn); % Plot measurements

%waitforbuttonpress;

% Initial weights are random unless we initialize random number generator
rng(12345, 'combRecursive');

% Create neural network
net = feedforwardnet(5, 'trainlm');

% Prepare train data set
samples_num = round(length(x) / samples_div) - (sample_length + result_length - 1);

sample = zeros(samples_num, sample_length);
result = zeros(samples_num, result_length);

for n = 1 : samples_num
    sample(n,:) = x(n:n + sample_length - 1);
    result(n,:) = x(n + sample_length:n + sample_length + result_length - 1);
end

% Plot training sample length
plot(t(1:n), x(1:n));

% Print generated test data
% fprintf("sample: "); disp(sample);
% fprintf("result: "); disp(result);

% Transpose test arrays to fit network inputs
sample = sample.';
result = result.';

% Configure network inputs:
net = configure(net, sample, result);
fprintf("net.inputs: %d\n", net.inputs{1}.size);

% Train network
net = train(net, sample, result);

% Test network
for n = 1 : test_step: length(xn) - (sample_length + result_length - 1)
    test_sample = xn(n:n + sample_length - 1);
    test_result = net(test_sample.').';

    % fprintf("test_sample: "); disp(test_sample);
    % fprintf("test_result: "); disp(test_result);

    % Plot network result
    sample_time = t(n):time_step:t(n + length(test_sample) - 1);
    result_time = t(n + sample_length):time_step:t(n + sample_length + length(test_result) - 1);

    plot(sample_time, test_sample, 'x'); waitforbuttonpress;
    plot(result_time, test_result, 'o'); waitforbuttonpress;
end

% Assess the performance of the trained network.
% The default performance function is mean squared error.
% perf = perform(net, x, x);

hold off;
