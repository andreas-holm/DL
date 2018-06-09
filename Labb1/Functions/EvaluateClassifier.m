%Returns the probabilities in matrix of a image being classified
function P = EvaluateClassifier(X, W, b)
    %Init
    %each column of X corresponds to an image and has size d*N
    
    labelSize = length(W(:,1));
    imageSize = length(X(1,:));
    %Each column contains probabilities for each label for a certain image
    %in the corresponding column of X. Size(P) = K*n (labelSize * sampleSize)
    P = zeros(labelSize, imageSize);
    %Evaluating
    for i= 1:imageSize
        s = W*X(:,i) + b;
        P(:,i) = exp(s)/sum(exp(s));
    end
end 