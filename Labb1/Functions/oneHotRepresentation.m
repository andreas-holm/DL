%yields the one-hot representation of y, a vector containing the labels for
%all samples
function [Y] = oneHotRepresentation(y)

    valueLabels = unique(y);
    numberOfLabels = length(valueLabels);
    numberOfSamples = length(y);

    Y = zeros(numberOfSamples, numberOfLabels);


    %one-hot representation means that the there

    for i = 1:numberOfLabels
        Y(:, i) = (y == valueLabels(i));     
    end

    Y = Y';

end