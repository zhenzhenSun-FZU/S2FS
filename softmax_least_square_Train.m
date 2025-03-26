function [optTheta] = softmax_least_square_Train(theta, inputData, labels,L, mu, options)
%softmaxTrain Train a softmax model with the given parameters on the given
% data. Returns softmaxOptTheta, a vector containing the trained parameters
% for the model.
% inputData: an N by M matrix containing the input data, such that
%            inputData(:, c) is the cth input
% labels: M by 1 matrix containing the class labels for the
%            corresponding inputs. labels(c) is the class label for
%            the cth input
% options (optional): options
%   options.maxIter: number of iterations to train for

if ~exist('options', 'var')
    options = struct;
end

if ~isfield(options, 'maxIter')
    options.maxIter = 400;
end

inputSize = size(inputData,1);
numClasses = size(labels,1);

% Use minFunc to minimize the function
addpath(genpath('minFunc'));
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % softmaxCost.m satisfies this.
minFuncOptions.display = 'on';
theta = reshape(theta, inputSize*numClasses, 1);     
[softmaxOptTheta, cost] = minFunc( @(p) softmax_least_square_cost(p, ...
                                   inputData, labels,L,mu), ...                                   
                              theta, options);

% Fold softmaxOptTheta into a nicer format
optTheta = reshape(softmaxOptTheta, inputSize, numClasses);                        
end                          
