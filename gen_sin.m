function [xr, xn] = gen_sin(t, A, w, phi, r, snr)
    if ~exist('r', 'var')
        r = 0;
    end

    noise_data = true;
    if ~exist('snr', 'var') || snr == 0
        noise_data = false;
    end

    xr = A .* sin(w * t + phi);
    xn = A .* (1 + normrnd(0, r)) .* sin(w * t + phi);

    if noise_data
        xn = awgn(xn, snr, 'measured');
    end

    [xr, xn] = rescale_data(xr, xn, 0, 1);
end
