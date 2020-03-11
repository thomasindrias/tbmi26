function [ scores ] = kFold(X, k, D, L)
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

numBins = 5;                    % Number of bins you want to divide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, ~, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);


%XBinComb = combineBins(XBins, [1,2,3]);

% Prepare a map of training data combinations (5 bins)
B = [1:5;1:5;1:5;1:5;1:5;];
B = B - diag(diag(B));
B = reshape(nonzeros(B'), size(B, 2)-1, [])';
% [2 3 4 5]
% [1 3 4 5]
% [1 2 4 5]
% [1 2 3 5]
% [1 2 3 4]

scores = zeros(k,1);

for ki = 1:k
    accuracies = 0;
    
    for n=1:numBins
        Xt = combineBins(XBins, B(n, :));
        Lt = combineBins(LBins, B(n, :));
        
        pred = kNN(XBins{n}, ki, Xt, Lt);
        cM = calcConfusionMatrix(pred, LBins{n});
        acc = calcAccuracy(cM);
        
        accuracies = accuracies + acc;
    end
    
    scores(ki) = accuracies/numBins;
end

end

