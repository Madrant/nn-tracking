function array = rand_loss(array, prob)
    ind = [];

    for n = 1:length(array)
        threshold = rand(1);

        if prob > threshold
            ind(end + 1) = n;
        end
    end

    array(ind) = [];
end
