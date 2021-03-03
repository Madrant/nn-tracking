clear all;

% Generate train data
t = 0:0.1:10;     % Time
x = sin(t);       % Coordinates
%x = awgn(x, 30); % Add some noise to reference data

% Add white gaussian noise to simulate measurements
tn = t;
xn = awgn(x, 20);

% Plot inputs
plot(t, x); % Plot reference data
hold on;
plot(tn, xn); % Plot measurements

% waitforbuttonpress;

% Construct a feedforward network with one hidden layer of size 10 neurons 
trainfunc = 'trainlm';
hiddenSizes = 10;
net = feedforwardnet(hiddenSizes, trainfunc);

% Configure generated network to best fit input data
net = configure(net, t, x);

% Plot untrained network output
% x1 = net(t);
% plot(t, x1, 'x');

% Train the network using the training data
net = train(net, t, x);

% View the trained network.
% view(net);

% Estimate the targets using the trained network.
t2 = 0:0.1:15;
x2 = net(t2);
plot(t2, x2, 'o');

% Assess the performance of the trained network.
% The default performance function is mean squared error.
% perf = perform(net, x1, x);

hold off;
