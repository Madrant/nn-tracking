% Source:
% https://stackoverflow.com/questions/61958087/designing-kalman-filter
function outputs = ts_kf(t, x, Y)
    A = 0.01;
    C = 1;

    % Covariance matrices
    % Processing noise
    W = 0.01;
    Q = 0.1; % A degree of trust to measurements

    % Measurement noise
    V = 1;
    R = 1;

    % Initial conditions
    x0 = x(1);
    P0 = 0;

    xpri = x0;
    Ppri = P0;

    xpost = x0;
    Ppost = P0;

    % State
    X = zeros(1, length(t));
    Xpri = zeros(1, length(t));
    
    X(1) = x0;
    Xpri(1) = x0;

    % xpri - x priori
    % xpost - x posteriori
    % Ppri - P priori
    % Ppost - P posteriori

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
            %xpri = A*xpost + Amp(i) * sin(w * t(i) + phi);
            xpri = A * xpost + x(i);
            Ppri = A * Ppost * A' + Q;

            eps = Y(i) - C * xpri;
            S = C * Ppri * C' + R;
            K = Ppri * C' * S^(-1);

            xpost = xpri + K * eps;
            Ppost = Ppri - K * S * K';

            Xpri(i) = xpri;
            X(i) = xpost;
        end
    end

    outputs = Xpri;
end

