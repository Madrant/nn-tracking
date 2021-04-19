% Get network output on test data set
function net_outputs = test_network(net, inputs, result_length)
    % Check input data
    assert(~isempty(inputs));
    assert(result_length > 0);

    samples_num = length(inputs);

    net_outputs = zeros(samples_num, result_length);

    for n = 1: samples_num
        input = inputs(n,:);

        output = net(input.');

        net_outputs(n,:) = output;
    end
    
    net_outputs = net_outputs.';
end

