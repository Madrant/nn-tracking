function cell_array = struct_fields_to_cell_array(struct_array, fields)
    cell_array = cell(length(struct), 1);
    array = zeros(length(fields), length(struct_array));

    for f = 1:length(fields)
        eval_str = sprintf("[struct_array(1:end).%s]", fields(f));
        data = eval(eval_str);

        array(f,:) = data;
    end

    cell_array{1} = array;
end
