function [X, Y, y] = LoadBatch(filename)
    scaling = 255;
    A = load(filename);
    
    %Load all picture data into a k * N dimensional matrix (k = 32 * 32 *
    %3), N = Number of samples
    X = double(A.data')/scaling;
    %vector of length N containing all labels
    y = A.labels;
    y = double(y) + ones(size(y));
    
    %one-hot representation of the label vector y, containing d * N
    %elements
    Y = oneHotRepresentation(y);
end 