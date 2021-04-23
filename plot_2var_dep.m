function plot_2var_dep(array, labels)
    x = unique(array(:,1));
    y = unique(array(:,2));
    z = array(:,3);

    [X, Y] = meshgrid(x, y);
    Z = reshape(z, size(X));

    figure;
    tiledlayout(1, 2);

    nexttile;
    surf(X, Y, Z);
    xlabel(labels(1));
    ylabel(labels(2));
    zlabel(labels(3));
    view(0, 180);

    nexttile;
    surf(X, Y, Z);
    xlabel(labels(1));
    ylabel(labels(2));
    zlabel(labels(3));
    view(90, 0);
end

