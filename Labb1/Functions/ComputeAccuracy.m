%calc the classifiers accuracy
function acc = ComputeAccuracy(X, y, W, b)
    %init
    acc = 0;
    P = EvaluateClassifier(X, W, b);
    %Get indexes for maximum prob which class for a certain image
    [maxProbabilities,index] = max(P);
    imageSize = length(X(1,:));
    
    for i=1:imageSize
        if(index(i) == y(i))
            acc =  acc + 1;
        end
    end
    
    acc = acc/imageSize;
end