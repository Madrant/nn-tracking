function layers = rnnBlock(hs_array, ht_array, act_array, dropout_array, numLayers)
    if ~exist('ht_array', 'var') || isempty(ht_array)
        ht_array = "lstm";
    end

    if ~exist('act_array', 'var') || isempty(act_array)
        act_array = "none";
    end

    if ~exist('numLayers', 'var')
        numLayers = 1;
    end

    if ~exist('dropout_array', 'var' )|| isempty(dropout_array)
        dropout_array = 0;
    end

    % Define layers array
    layers = lstmLayer(10);

    % Reproduce parameters arrays if numLayers is larger than perameters list
    if length(hs_array) < numLayers
        hs_array = [hs_array repmat(hs_array(1, end), 1, numLayers - length(hs_array))];
    end

    if length(ht_array) < numLayers
        ht_array = [ht_array repmat(ht_array(1, end), 1, numLayers - length(ht_array))];
    end

    if length(act_array) < numLayers
        act_array = [act_array repmat(act_array(1, end), 1, numLayers - length(act_array))];
    end

    if length(dropout_array) < numLayers
        dropout_array = [dropout_array repmat(dropout_array(1, end), 1, numLayers - length(dropout_array))];
    end

    for layer = 1:numLayers
        hiddenSize = hs_array(1,layer);
        hiddenType = ht_array(1,layer);
        activation = act_array(1,layer);
        dropout = dropout_array(1, layer);

        fprintf("Type: %s Hidden size: %u Activation: %s Dropout: %f\n", hiddenType, hiddenSize, activation, dropout);

        assert(hiddenSize > 0);
        assert(hiddenType == "lstm" || hiddenType == "gru");
        assert(activation == "relu" || activation == "tanh" || activation == "none" || activation == "leakedrelu");
        assert(dropout >= 0.0 && dropout <= 1.0);

        if hiddenType == "gru"
            if layer == 1
                layers(end,:) = gruLayer(hiddenSize);
            else
                layers(end + 1,:) = gruLayer(hiddenSize);
            end
        elseif hiddenType == "lstm"
            if layer == 1
                layers(end,:) = lstmLayer(hiddenSize);
            else
                layers(end + 1,:) = lstmLayer(hiddenSize);
            end
        end

        % Determine activation type
        if activation == "relu"
            layers(end + 1,:) = reluLayer;
        elseif activation == "leakedrelu"
            layers(end + 1,:) = leakyReluLayer;
        elseif activation == "tanh"
            layers(end + 1,:) = tanhLayer;
        end

        if (dropout > 0)
            layers(end + 1,:) = dropoutLayer(dropout);
        end
    end
end
