function acc = Mysoftmax_classifier
%% STEP 0: Read the Data Set and labels
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
labels(labels==0) = 10; 
inputData = images;
%% STEP 0: The Data Set is divided into train set and test set
nsample = floor(size(inputData,2)*0.8);   
DataTrain = inputData(:,1:nsample); % The top 80% datas and labels as the train set
labelsTrain = labels(1:nsample);
DataTest = inputData(:,nsample+1:end); % The last 20% as the test set
labelsTest = labels(nsample+1:end);
%% STEP 1: Initialise constants and parameters
inputSize = size(inputData,1); % Size of input vector (MNIST images are 28x28)
numClasses = 10;     % Number of classes (MNIST images fall into 10 classes)
lambda = 1e-4;       % Weight decay parameter
theta = 0.005 * randn(numClasses * inputSize, 1);   % Randomly initialise theta 
%% STEP 2: Implement softmaxCost 
% [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, inputData, labels);

%% STEP 3: Gradient checking
% numGrad = computeNumericalGradient( @(x) softmaxCost(x, numClasses,inputSize,...
%                                                lambda, inputData, labels), theta);
% diff = norm(numGrad-grad)/norm(numGrad+grad);
%% STEP 4: Learning parameters
options.maxIter = 200;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            DataTrain, labelsTrain, options);    
%% STEP 5: Testing
[pred] = softmaxPredict(softmaxModel, DataTest);
acc = mean(labelsTest(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);
