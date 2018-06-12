format long

%init model 

% Data -----------------------------------------

[XtrainSet, YtrainSet, ytrainSet] = LoadBatch('data_batch_1.mat');

[XtestSet, YtestSet, ytestSet] = LoadBatch('test_batch.mat');
[XvalSet, YtestSet, ytestSet] = LoadBatch('test_batch.mat');

%Parameters-------------------------------------
%Init each entry to have Gaussian dist random values
meanInit = 0;
stdInit = 0.1;

%number of mini-batches to divide the whole dataset into
n_batch = 100;
%Learning rate
eta = 0.1;
%number of epochs to train for
n_epochs = 40;
%regularization penalization term lambda
lambda = 0;

%-----------------------------------------------
GradientDescent_params = [n_batch eta n_epochs];
%Init Matrix and bias
[W, b] = Init(stdInit, meanInit, Xtrain, Ytrain);


sampleSize = length(Xtrain(1,:));
%ending = int64(part*sampleSize);
 %XtrainSet = Xtrain(:, 1:ending);
 %YtrainSet = Ytrain(:, 1:ending);
 %ytrainSet = ytrain(1:ending);
 
%Genereate our validation set
 %XvalSet = Xtrain(:, (ending + 1):end);
 %YvalSet = Ytrain(:, (ending + 1):end);


J_InitCost = ComputeCost(XtrainSet, YtrainSet, W, b, lambda)
J_BestValCost = ComputeCost(XvalSet, YvalSet, W, b, lambda);
J_TrainingCostVector = [];
J_valCostVector = [];

Wstar = W;
bstar = b;
%Training Wstar and bstar
for i = 1:n_epochs
    
    [WstarTemp, bstarTemp] = MiniBatchGD(XtrainSet, YtrainSet, GradientDescent_params, Wstar, bstar, lambda);
    i
    J_TrainingCost = ComputeCost(XtrainSet, YtrainSet, WstarTemp, bstarTemp, lambda)
    J_validCost = ComputeCost(XvalSet, YvalSet, WstarTemp, bstarTemp, lambda);
    J_TrainingCostVector = [J_TrainingCostVector J_TrainingCost];
    
    if(J_validCost < J_BestValCost)
        Wstar = WstarTemp;
        bstar = bstarTemp;
        J_BestValCost = J_validCost;
    end
    
    GradientDescent_params(2) = 0.9*GradientDescent_params(2);
    
    J_valCost = ComputeCost(XvalSet, YvalSet, Wstar, bstar, lambda)
    J_valCostVector = [J_valCostVector J_valCost];
end 
trainingCostInit  = ComputeCost(Xtrain, Ytrain, W, b, lambda)
finalTrainCost = ComputeCost(Xtrain, Ytrain, Wstar, bstar, lambda)
trainingAccInit = ComputeAccuracy(XtrainSet, ytrainSet, W, b)
trainingAcc = ComputeAccuracy(XtrainSet, ytrainSet, Wstar, bstar)
testAcc = ComputeAccuracy(Xtest, ytest, Wstar, bstar)

plotCostFunctions(J_TrainingCostVector, J_valCostVector, n_epochs);
hold on
%DisplayWMatrix(Wstar);

