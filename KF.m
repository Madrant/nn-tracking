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
Q = 0.05; % A degree of trust to measurements

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
Xpri = Y;
X(1) = x0;
Xpri(1) = x0;

% xpri - x priori
% xpost - x posteriori
% Ppri - P priori
% Ppost - P posteriori

w = 1 * pi;
phi = 0;
Amp = floor(t);
snr = 20;

rng(12345, 'combRecursive');
r = 0.01;
x = Amp .* (1 + normrnd(0, r)) .* sin(w * (1 + normrnd(0, r)) * t + 1.2 * phi*(1 + normrnd(0, r)));
%x = Amp .* sin(w * t + phi);
Y = awgn(x, snr, 'measured');

for i = 1:length(t)
    % Generate real data
    %x(i) = C*sin((i-1) * 0.1);
    %x(i) = Amp(i) * (1 + normrnd(0, 0.01)) * sin(w * (1 + normrnd(0, 0.01)) * t(i) + phi * (1 + normrnd(0, 0.01)));

    % Noise measurements
    %Y(i) = x(i) + normrnd(0, sqrt(V));
    %Y = awgn(x, snr);

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
        
        Xpri(i) = xpri;
        X(i) = xpost;
    end
end

% Calculate absolute errors
%error = (x(1:end - 1) - X(2:end));

plot_results("KF", t, x, Y, Xpri, false, 0);
