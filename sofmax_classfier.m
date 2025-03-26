function [ acc,pred,score] = sofmax_classfier( DataTrain, labelsTrain, DataTest, labelsTest)
inputSize = size(DataTrain,1); % Size of input vector 
numClasses = max(max(labelsTrain),max(labelsTest));    % Number of classes 
lambda = 1e-4;       % Weight decay parameter
%% STEP 4: Learning parameters
options.maxIter = 200;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            DataTrain, labelsTrain, options);    
%% STEP 5: Testing
[pred,score] = softmaxPredict(softmaxModel, DataTest);
acc = mean(labelsTest(:) == pred(:));
%fprintf('Accuracy: %0.3f%%\n', acc * 100);
end



