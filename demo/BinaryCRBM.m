%fprintf('\nHere we train a Convolutional RBM on the MNIST Dataset.\n');
%dataset = 'mnistSmall';

%[images, tags] load('data.mat');
testRun = 2;
[images, tags, m, n] = loadImagesTags;
dataSize = [m,n,1];  % [nY x nX x nChannels]

trainTag = cell(1,1);
testTag = cell(1,1);
experiments = size(images);
trainImgCount = 1;
testImgCount = 1;
for i = 1 : experiments(1)
    experimentSize = size(images{i,1});
    if i ~= testRun
        for j = 1 : experimentSize(1)
            trainData(:,:,trainImgCount) = images{i,1}{j,2};
            trainTag{trainImgCount,1} = images{i,1}{j,3};
            trainImgCount = trainImgCount + 1;
        end
    else
        for j = 1 : experimentSize(1)
            testData(:,:,testImgCount) = images{i,1}{j,2};
            testTag{testImgCount,1} = images{i,1}{j,3};
            testImgCount = testImgCount + 1;
        end
    end
end

% DEFINE AN ARCHITECTURE
arch = struct('dataSize', dataSize, ...
		'nFM', 9, ...
        'filterSize', [7 7], ...
        'stride', [2 2], ...
        'inputType', 'binary');

% GLOBAL OPTIONS
arch.opts = {'nEpoch', 1, ...
			 'lRate', .00001, ...
			 'displayEvery',100, ...
			 'sparsity', .02, ...
			 'sparseGain', 5};% , ...
%  			 'visFun', @visBinaryCRBMLearning}; % UNCOMMENT TO VIEW LEARNING

% INITIALIZE AND TRAIN
cr = crbm(arch);
cr = cr.train(trainData);

for i = 1 : (trainImgCount - 1)
    % INFER HIDDEN AND POOLING LAYER EXPECTATIONS
    % CONDITIONED ON SOME INPUT
    [cr,ep] = cr.poolGivVis(trainData(:,:,i));
    cr = cr.hidGivVis(trainData(:,:,i));

    [nCols,nRows,k]=size(cr.eHid);
    eh = reshape(cr.eHid,nRows*nCols*k,1);
    trainFeatures(i,:) = eh;
end

% INITIALIZE AND TRAIN
cr = crbm(arch);
cr = cr.train(testData);

for i = 1 : (testImgCount - 1)
    % INFER HIDDEN AND POOLING LAYER EXPECTATIONS
    % CONDITIONED ON SOME INPUT
    [cr,ep] = cr.poolGivVis(testData(:,:,i));
    cr = cr.hidGivVis(testData(:,:,i));

    [nCols,nRows,k]=size(cr.eHid);
    eh = reshape(cr.eHid,nRows*nCols*k,1);
    testFeatures(i,:) = eh;
end

models = cell(length(tags),1);
% SVM Training
for i = 1 : length(tags)
    trainLabel = ones((trainImgCount - 1), 1)*-1;
    for j = 1 : (trainImgCount - 1)
        if ismember(tags{i,1}, trainTag{j,1})
            trainLabel(j,1) = 1;
        end
    end
    models{i,1} = svmtrain(trainLabel,trainFeatures);
end

% Predict
result = cell(length(tags),2);
for i = 1 : length(tags)
    testLabel = ones((testImgCount - 1), 1)*-1;
    for j = 1 : (testImgCount - 1)
        if ismember(tags{i,1}, testTag{j,1})
            testLabel(j,1) = 1;
        end
    end
    [predict_label, accuracy, prob_estimates] = svmpredict(testLabel,testFeatures,models{i,1},'-b 1');
    result{i,1} = [predict_label, prob_estimates, testLabel, tags{i,1}*ones((testImgCount - 1), 1)];
    result{i,2} = accuracy;
end