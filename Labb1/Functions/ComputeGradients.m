function ComputeGradients(X, Y, W, b, lambda)
    imageSize = length(X(1,:));
    
    %calc more accurate version of gradient based on the centered difference formula
    [ngrad_b, ngrad_W] = ComputeGradsNumSlow(X(:,1), Y(:,1), W, b, lambda, 1e-6);
    
    %compute the gradient for the first 100 dimensions of the input
    %[ngrad_bTrick, ngrad_WTrick] = ComputeGradsNumSlow(X(1:100, 1), Y(:,1), W(:,100), b, lambda, 1e-6);
    
    
    %a faster but less accurate based on the infite difference method
    [agrad_b, agrad_W] = ComputeGradsAnalytical(X(:,1), Y(:,1), W, b, lambda);
    
    
    %relative error for b and W
    epsilon = 1e-20;
    diffW = ngrad_W - agrad_W;
    %relError_b = abs(ngrad_b - agrad_b)/max(epsilon, abs(abs(ngrad_b) + abs(agrad_b))));
    %relError_W = abs(diffW)/max(epsilon, abs(abs(ngrad_W) + abs(agrad_W)))
    
    relErrorLimit = 1e-6;
%     %check if difference is good enough
%     if (max(max(abs(relError_b))) < relErrorLimit) && (max(max(abs(relError_W))) < relErrorLimit)
%         disp('Correct! Using as error limit ')
%         disp(relErrorLimit)
%         disp('with error ')
%         disp(max(max(abs(relError_b))))
%         disp(' ')
%         disp(max(max(abs(relError_W))))
%     else
%         disp('Wrong! Using as error limit ')
%         disp(relErrorLimit)
%         disp('with error ')
%         disp(max(max(abs(relError_b))))
%         disp(' ')
%         disp(max(max(abs(relError_W))))
%     end
    
    grad_W = agrad_W;
    grad_b = agrad_b;    
    
    
end