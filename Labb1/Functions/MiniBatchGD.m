
function [Wstar, bstar] = MiniBatchGD(Xtrain, Ytrain, GradientDescent_params, W, b, lambda)
    
    n_batch = GradientDescent_params(1);
    eta = GradientDescent_params(2);
    Wstar = W;
    bstar = b;
    
    sampleSize = length(Xtrain(1,:));
    for j = 1:sampleSize/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j * n_batch;
        
        Xbatch = Xtrain(:,j_start:j_end);
        Ybatch = Ytrain(:,j_start:j_end);    
        [grad_b, grad_W] = ComputeGradsAnalytical(Xbatch, Ybatch, Wstar, bstar, lambda);
        
        Wstar = Wstar - eta.*grad_W;
        bstar = bstar - eta.*grad_b; 
    end
end
