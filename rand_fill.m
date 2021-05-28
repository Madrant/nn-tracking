function array = rand_fill(array, value, prob)
    array = arrayfun(@(array_value)rand_set(array_value, value, prob), array);
end

function ret = rand_set(array_value, value, prob)
    threshold = rand(1);

    if prob > threshold
        ret = value;
    else
        ret = array_value;
    end
end
