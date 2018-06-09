function [ W, b] = Init(stdInit, meanInit, Xdata, Ydata)
%Init matrix and bias with given init std and mean

labelSize = length(Ydata(:,1));
imageSize = length(Xdata(:,1));
rng(400);
Winit = randn(labelSize, imageSize);
bInit = randn(labelSize, 1);

W = meanInit + stdInit*Winit;
b = meanInit + stdInit*bInit;

end

