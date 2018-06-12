function plotCostFunctions(trainCost, validationCost, n_epochs)
    %plot the cost computed for each epoch
    plot(1:n_epochs, trainCost)
    hold on
    plot(1:n_epochs, validationCost)
    xlabel('Epochs')
    xlabel('Cost')
    legend('Training cost','Validation cost')

end