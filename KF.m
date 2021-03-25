% Source:
% https://stackoverflow.com/questions/61958087/designing-kalman-filter
clear all;
close all;
clc;

t = 0:0.1:10;

A = 0.01;
C = 1;

% Covariance matrices
% Processing noise
W = 0.01;
Q = 0.01; % A degree of trust to measurements

% Measurement noise
V = 1;
R = 1;

% Initial conditions
x0 = 0;
P0 = 0;

xpri = x0;
Ppri = P0;

xpost = x0;
Ppost = P0;

% State
x = zeros(1, length(t));
Y = zeros(1, length(t));
X = Y;
X(1) = x0;

% xpri - x priori
% xpost - x posteriori
% Ppri - P priori
% Ppost - P posteriori

w = 1 * pi;
phi = 0;
Amp = floor(t);
snr = 5;

for i = 1:length(t)
    % Generate real data
    %x(i) = C*sin((i-1) * 0.1);
    x(i) = 1.2 * Amp(i) * sin(w * t(i) + phi);

    % Noise measurements
    %Y(i) = x(i) + normrnd(0, sqrt(V));
    Y(i) = awgn(x(i), snr);

    if i > 1
        % Prediction
        % xpri = A*xpost + Amp(i)*sin((i-1)*0.1);
        xpri = A*xpost + Amp(i) * sin(w * t(i) + phi);
        Ppri = A*Ppost * A' + Q;

        eps = Y(i) - C * xpri;
        S = C * Ppri * C' + R;
        K = Ppri * C' * S^(-1);

        xpost = xpri + K * eps;
        Ppost = Ppri - K * S * K';

        X(i) = xpost;
    end
end

% Calculate absolute errors
error = (Y - X);
abs_error = abs(error);
max_error = max(abs_error);
mse = mean(error.^2);
rmse = sqrt(mse);

% Calculate MSE for time series
mse_array = zeros(length(error));
rmse_array = zeros(length(error));

for n=1:length(error)
    mse = mean(error(1:n).^2);
    mse_array(n) = mse;
    rmse_array(n) = sqrt(mse);
end

fprintf("Max error: %f\n", max_error);
fprintf("MSE:       %f\n", mse);
fprintf("RMSE:      %f\n", rmse);

plot(t, x, 'gx', t, Y, 'b', t, X, 'r')
title('Kalman Filter')
xlabel('Time')
ylabel('Value')
legend('Real value', 'Measured value', 'Estimated value')
