format long

%init model 

[Xtrain, Ytrain, ytrain] = LoadBatch('data_batch_1.mat');

[Xtest, Ytest, ytest] = LoadBatch('test_batch.mat');

%Size(b) = #Labels * 1
%Size(W) = #Labels * dim of each image

%Init each entry to have Gaussian dist random values
meanInit = 0;
stdInit = 0.1;

%Init Matrix and bias
[W, b] = Init(stdInit, meanInit, Xtrain, Ytrain);

