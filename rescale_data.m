function [x1, x2] = rescale_data(x1, x2, from, to)
    rescale_array = rescale([x1; x2], from, to);
    x1 = rescale_array(1,:);
    x2 = rescale_array(2,:);
end

