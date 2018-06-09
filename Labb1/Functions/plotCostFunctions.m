function plotCostFunctions(trainCost, validationCost, n_epochs)
    %plot the cost computed for each epoch
    plot(1:n_epochs, J_costTrain_vector)
    hold on
    plot(1:n_epochs, J_costValidation_vector)
    xlabel('Epochs')
    xlabel('Cost')
    legend('Training cost','Validation cost')

end