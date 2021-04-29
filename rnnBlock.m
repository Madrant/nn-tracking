function layers = rnnBlock(hiddenSize, hiddenType, activation, numLayers, dropout)
    assert(hiddenType == "lstm" || hiddenType == "gru");
    assert(activation == "relu" || activation == "tanh" || activation == "none" || activation == "leakedrelu");

    if ~exist('hiddenType', 'var')
        hiddenType = "lstm";
    end

    if ~exist('numLayers', 'var')
        numLayers = 1;
    end

    if ~exist('dropout', 'var')
        dropout = 0;
    end

    % Determine main working layer type
    layer = [lstmLayer(hiddenSize)];

    if hiddenType == "gru"
        layer(end,:) = gruLayer(hiddenSize);
    end

    % Determine activation type
    if activation == "relu"
        layer(end + 1,:) = reluLayer;
    elseif activation == "leakedrelu"
        layer(end + 1,:) = leakyReluLayer;
    elseif activation == "tanh"
        layer(end + 1,:) = tanhLayer;
    end

    if (dropout > 0)
        layer(end + 1,:) = dropoutLayer(dropout);
    end

    % Copy layers 'numLayers' time
    layers = repmat(layer, numLayers, 1);
end