function [grad_b, grad_W] = ComputeGradsAnalytical(X, Y, W, b, lambda)
    

    imageDimensions = length(X(:,1));
    P = EvaluateClassifier(X, W, b);
    
    grad_b = zeros(length(Y(:,1)),1);
    grad_W = zeros(length(Y(:,1)), imageDimensions);
    size(grad_W)
    imageSize = length(X(1,:));
    for i = 1:imageSize
        probVec = P(:,i);
        yVec = Y(:,i);
        yVecT = yVec';
        xVec= X(:,i);
        
        grad = - (yVec/(yVecT*probVec))*(diag(probVec) - probVec*probVec');
        gradT = grad';
        size(grad)
        grad_b = grad_b + gradT;
        grad_W = grad_W + gradT*xVec';
    end
    
    grad_W_norm = grad_W/imageSize;
    grad_b_norm = grad_b/imageSize;
   
    grad_W = grad_W_norm + 2*lambda*W;
    grad_b = grad_b_norm;

end