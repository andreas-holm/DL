function [J] = ComputeCost(X, Y, W, b, lambda)
   %size of a image    
    sampleSize = size(X,2);
    
    %Calc classification probabilties
    P = EvaluateClassifier(X, W, b);
    
    %calc cross-entropy
    crossEntropy = sum(sum(-log(P(logical(Y)))));
    
    %calc regularization
    reg = lambda*sum(sum(W.^2));
    
    %calc cost-function
    J = (1/sampleSize)*crossEntropy + reg;
end