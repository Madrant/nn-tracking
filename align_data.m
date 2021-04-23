function [am, bm] = align_data(a, b)
    am = a;
    bm = b;

    if length(a) > length(b)
        diff = length(a) - length(b) + 1;
        am = a(diff:length(a));
    elseif length(b) > length(a)
        diff = length(b) - length(a) + 1;
        bm = b(diff:length(b));
    end
end
