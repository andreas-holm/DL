function [grad_b, grad_W] = ComputeGradsAnalytical(X, Y, W, b, lambda)
    

    imageDimensions = length(X(:,1));
    P = EvaluateClassifier(X, W, b);
    sampleSize = length(X(1,:));
    imageSize = length(Y(:,1));
    grad_b = zeros(imageSize, 1);
    grad_W = zeros(imageSize, imageDimensions);
    
    for i = 1:sampleSize
        y = Y(:,i);
        x= X(:,i);
        p = P(:,i);
        g = - (y'/(y'*p)) * (diag(p) - p*p');
        grad_b = grad_b + g';
        grad_W = grad_W + g'*x';
    end
    
    grad_W_norm = grad_W/sampleSize;
    grad_b_norm = grad_b/sampleSize;
   
    grad_W = grad_W_norm + 2*lambda*W;
    grad_b = grad_b_norm;

end