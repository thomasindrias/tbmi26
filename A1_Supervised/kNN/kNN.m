function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)
%
% X - Xtrain index 2934 

classes = unique(LTrain);
NClasses = length(classes);

% Add your own code here
dist = pdist2(X,XTrain);
[~, I] = sort(dist, 2);

% Get the closest neighbour and get the predicted
% label from that pos.
pos = I(:, 1:k);
weight = LTrain(pos);

% Classify
LPred = mode(weight, 2);

end

