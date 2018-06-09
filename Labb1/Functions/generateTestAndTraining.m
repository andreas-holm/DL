function [Xtrain, Ytrain, ytrain, Xvalid, Yvalid, yvalid] = generateTestAndTraining(X, Y, y, partTraining)
    
    sampleSize = length(X(1,:));
    trainingEndIndex = int64(partTraining * sampleSize);  
    %validationEndIndex = int64((partTraining +  partValidation) * sampleSize);
    Xtrain = X(:, 1:trainingEndIndex);
    Ytrain = Y(:, 1:trainingEndIndex);
    ytrain = y(1:trainingEndIndex);
    Xvalid = X(:, (trainingEndIndex + 1):end);
    Yvalid = Y(:, (trainingEndIndex + 1):end);
    yvalid = y(:, (trainingEndIndex + 1):end);
    
end