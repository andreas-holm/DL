format long

%init model 

[Xtrain, Ytrain, ytrain] = LoadBatch('data_batch_1.mat');

[Xtest, Ytest, ytest] = LoadBatch('test_batch.mat');

%Size(b) = #Labels * 1
%Size(W) = #Labels * dim of each image

%Init each entry to have Gaussian dist random values
meanInit = 0;
stdInit = 0.1;

%number of mini-batches to divide the whole dataset into
n_batch = 100;
%the learning rate
eta = 0.05;
%number of epochs to train for
n_epochs = 100;
%regularization penalization term lambda
lambda = 0;


GradientDescent_params = [n_batch eta n_epochs];
%Init Matrix and bias
[W, b] = Init(stdInit, meanInit, Xtrain, Ytrain);

%generate training, validation and test sets
partTraining = 0.9;
%partValidation = 1 - partTraining;
[Xtrain, Ytrain, ytrain, Xvalid, Yvalid] = generateTestAndTraining(Xtrain, Ytrain, ytrain, partTraining);


J_cost_init = ComputeCost(Xtrain, Ytrain, W, b, lambda)
J_costValidBest = ComputeCost(Xvalid, Yvalid, W, b, lambda);
J_costTrain_vector = [];
J_costValidation_vector = [];

Wstar = W;
bstar = b;
%learn Wstar and bstar matrices
for i = 1:n_epochs
    
    [WstarIter, bstarIter] = MiniBatchGD(Xtrain, Ytrain, GradientDescent_params, Wstar, bstar, lambda);
    i
    J_costTrain = ComputeCost(Xtrain, Ytrain, WstarIter, bstarIter, lambda);
    J_costValid = ComputeCost(Xvalid, Yvalid, WstarIter, bstarIter, lambda);
    J_costTrain_vector = [J_costTrain_vector J_costTrain];
    
    if(J_costValid < J_costValidBest)
        Wstar = WstarIter;
        bstar = bstarIter;
        J_costValidBest = J_costValid
    end
    
    %set a decay rate 
    GradientDescent_params(2) = 0.9*GradientDescent_params(2);
    
    %J_costValidation = ComputeCost(Xvalid, Yvalid, Wstar, bstar, lambda)
    %J_costValidation_vector = [J_costValidation_vector J_costValidation];
    
end 
startTrainAccuracy = ComputeAccuracy(Xtrain, ytrain, W, b)
finalTrainAccuracy = ComputeAccuracy(Xtrain, ytrain, Wstar, bstar)
finalTestAccuracy = ComputeAccuracy(Xtest, ytest, Wstar, bstar)

plotCostFunctions(J_costTrain_vector, J_costValidation_vector, n_epochs);

