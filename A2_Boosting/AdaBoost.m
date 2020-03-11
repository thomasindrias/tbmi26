%% Hyper-parameters
clear
% Number of randomized Haar-features
nbrHaarFeatures = 300;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 1000;
% Number of weak classifiers
nbrWeakClassifiers = 100;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError

d = ones(nbrTrainImages, 1)./nbrTrainImages;
tpfa = zeros(nbrWeakClassifiers,4);

for c = 1:nbrWeakClassifiers
    e_min = inf;
    t_min = 0;
    p_min = 0;
    f_min = 0;
    a_min = 0;
    h = 0;
    
    for i=1:nbrHaarFeatures
        for j=1:nbrTrainImages
            polarity = 1.0;
            
            %Small decimal value so we dont divide by zero for d
            t = xTrain(i,j) + 0.0001;
            hi = WeakClassifier(t, polarity, xTrain(i,:));
            e = WeakClassifierError(hi,d,yTrain);

            % Change Polarity
            if e > 0.5
                polarity = -1.0;
                e = 1-e;
            end

            if e < e_min
                e_min = e;
                alpha = 0.5 * log((1-e_min) / e_min);
                t_min = t;
                p_min = polarity;
                f_min = i;
                a_min = alpha;
                h = polarity*hi;
            end
        end
    end
    
    d = d .* exp(-a_min * yTrain .* h)';
    d = d ./ sum(d);
    tpfa(c,1) = t_min;
    tpfa(c,2) = p_min;
    tpfa(c,3) = f_min;
    tpfa(c,4) = a_min;
end
    
%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

classifier = zeros(nbrWeakClassifiers, size(yTest,2));
for i = 1:nbrWeakClassifiers
    classifier(i,:) = tpfa(i,4) * WeakClassifier(tpfa(i, 1), tpfa(i, 2), xTest(tpfa(i, 3),:));
    c = sum(classifier(1:i,:),1);
    c(c>0) = 1;
    c(c<0) = -1;
    acc(i) = 1 - mean(abs(c - yTest))/2;
end


%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

plot(1:nbrWeakClassifiers,acc)

%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.

figure(1);
colormap gray;
i = 1;
k = 1;
while k < 25
 
    if c(i) ~= yTest(i)
        subplot(5,5,k), imagesc(testImages(:,:,i));
        title("cls: " + c(i) + " yTest: " + yTest(i))  
        axis image;
        axis off;
        k = k + 1;
    end
    i = i + 1;
end


figure(2);
colormap gray;
i = nbrTestImages;
k = 1;
while k < 25
 
    if c(i) ~= yTest(i)
        subplot(5,5,k), imagesc(testImages(:,:,i));
        title("cls: " + c(i) + " yTest: " + yTest(i))
        axis image;
        axis off;
        k = k + 1;
    end
    i = i - 1;
end

%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.
figure(1);
colormap gray;
k = 1;
for i = 1:nbrWeakClassifiers
    if k > 25
        break
    end
    
    subplot(5,5,k), imagesc(haarFeatureMasks(:,:,tpfa(i, 3)));
    axis image;
    axis off;
    k = k+1;

end