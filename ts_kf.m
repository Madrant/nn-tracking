% Source:
% https://stackoverflow.com/questions/61958087/designing-kalman-filter
%
% See also:
% https://en.wikipedia.org/wiki/Kalman_filter
function outputs = ts_kf(t, x, Y, predict_offset)
    assert(length(t) == length(x));
    assert(length(x) == length(Y));

    if ~exist('predict_offset', 'var')
        predict_offset = 0;
    end
    
    % Process descritption
    F = 0.1; % Process state transition model (How much xpri depends of previous xpost)
    B = 1; % Process control model (How much target controls affects measurements model)

    % Measurements description
    H = 1; % Observation model (Coefficient to innovation - a difference between observed and predicted state)

    % Covariance matrices
    Q = 0.8; % Process noise covariance (A degree of trust to measurements)
    R = 1;   % Measurements (observations) noise covariance

    % Define state arrays
    X = zeros(1, length(t));
    Xpri = zeros(1, length(t));

    % For each measurement
    for n = 1:length(t)
        % Start filtering after receiving the first measurement
        if n <= 1
            xpost = x(n);
            Ppost = 0;

            X(n) = x(n);
            Xpri(n) = x(n);

            continue;
        end

        % Prediction
        %
        % Predict state
        if predict_offset >= 1
            xpri = F * xpost + B * x(n + predict_offset - 1);
        else
            xpri = F * xpost + B * x(n);
        end

        % Predict estimate covariance
        Ppri = F * Ppost * F' + Q;

        % Innovation (pre-fit residual): a difference between observed and predicted state
        eps = Y(n) - H * xpri;

        % Calculate innovation (pre-fit residual) covariance
        S = H * Ppri * H' + R;

        % Calculate Kalman gain
        K = Ppri * H' * S^(-1);

        % Update
        %
        % Update state
        xpost = xpri + K * eps;

        % Update estimate covariance
        Ppost = Ppri - K * S * K';

        % Save prior and a posterior state to arrays
        Xpri(n) = xpri;
        X(n) = xpost;
    end

    % Return predicted or filtered values based on 'predict_offset' value
    if predict_offset >= 1
        outputs = Xpri;
    else
        outputs = X;
    end
end
