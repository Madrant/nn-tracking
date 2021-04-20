% Get network output on test data set
function net_outputs = test_network(net, inputs, result_length, nn_type)
    if ~exist('result_length', 'var')
        result_length = 1;
    end

    if ~exist('nn_type', 'var')
        nn_type = 'ff';
    end

    assert(~isempty(inputs))
    
    % Check input data
    assert(~isempty(inputs));
    assert(result_length > 0);

    samples_num = length(inputs);

    net_outputs = zeros(samples_num, result_length);

    for n = 1: samples_num
        input = inputs(n,:);

        if nn_type == "ff"
            output = net(input.');
        elseif nn_type == "lstm" || nn_type == "gru"
            [net, output] = predictAndUpdateState(net, input.');
        end

        net_outputs(n,:) = output;
    end

    net_outputs = net_outputs.';
end

