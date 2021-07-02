function [input, output] = create_train_data_set(data, predict_offset, samples_div, obs_period, obs_duration, obs_start, loss_array, snr_array)
    input = cell(0);
    output = cell(0);

    for loss_prob = loss_array
    for snr = snr_array

        train_data = prepare_train_data(...
            data, predict_offset, samples_div, ...
            obs_period, obs_duration, obs_start, ...
            loss_prob, snr);

        input(:,end + 1) = struct_fields_to_cell_array(train_data, ["dt" "xn"]).';
        output(:,end + 1) = struct_fields_to_cell_array(train_data, ["xr"]).';
    end % snr
    end % loss_prob
end
