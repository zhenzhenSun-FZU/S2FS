function [cost, grad] = softmax_least_square_cost(theta, data, labels, L, mu)
% INPUT:
%      * lambda - weight decay parameter
%      * data - the N x M input matrix, where each column data(:, i) corresponds to
%               a single test set
%      * labels - an M x 1 matrix containing the labels corresponding for the input data
% OUTPUT:
%      * cost - 
%      * grad - 

% Unroll the parameters from theta
inputSize = size(data,1);
numClasses = size(labels,1);
numCases = size(data, 2);
theta = reshape(theta, inputSize, numClasses);   

cost = 0;
thetagrad = zeros(inputSize, numClasses);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
M = theta'*data;
M = bsxfun(@minus,M,max(M, [], 1));   
M = exp(M);
p = bsxfun(@rdivide, M, sum(M));
cost = -1/numCases * labels(:)' * log(p(:)) +mu/2 * sum((theta(:)-L(:)) .^ 2);
thetagrad = -1/numCases * data * (labels - p)' + mu * (theta-L);

% M = theta * data;
% M = bsxfun(@minus, M, max(M, [], 1));
% p = exp(M) ./ repmat(sum(exp(M)), numClasses, 1);
% cost = -(1. / numCases) * sum(sum(groundTruth .* log(p))) + (lambda / 2.) * sum(sum(theta.^2));
% thetagrad = -(1. / numCases) * (groundTruth - p) * data' + lambda * theta;
% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

