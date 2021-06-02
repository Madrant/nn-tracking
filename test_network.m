% Get network output on test data set
function net_outputs = test_network(net, input_cell_array, num_outputs, nn_type)
    if ~exist('nn_type', 'var')
        nn_type = 'ff';
    end

    % Check input data
    assert(~isempty(input_cell_array));

    inputs = input_cell_array{1}();

    disp(size(inputs));

    num_features = size(inputs, 1);
    samples_num = size(inputs, 2);

    fprintf("num_features: %u\n", num_features);
    fprintf("samples_num: %u\n", samples_num);

    net_outputs = zeros(samples_num, num_outputs);

    for n = 1: samples_num
        input = inputs(1:end, n);

        if nn_type == "ff"
            output = net(input.');
        elseif nn_type == "lstm" || nn_type == "gru"
            [net, output] = predictAndUpdateState(net, input, 'ExecutionEnvironment', 'cpu');
        end

        net_outputs(n,:) = output;
    end

    net_outputs = net_outputs.';
end
