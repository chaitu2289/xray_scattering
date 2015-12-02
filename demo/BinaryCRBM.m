
dataSet = load('data.mat');
images = dataSet.images;
tags = dataSet.tags;
clear dataSet;

dataSize = [256,256,1];  % [nY x nX x nChannels]
imgCount = 1;
experiments = size(images);
Data = [];
expToImg = cell(1,1);
for i = 1 : experiments(1)
    experimentSize = size(images{i,1});
    expToImg{i,1} =  experimentSize(1);
    for j = 1 : experimentSize(1)
        Data(:,:,imgCount) = images{i,1}{j,2};
	images{i,1}{j,2} = [];
        tag{imgCount,1} = images{i,1}{j,3};
        imgCount = imgCount + 1;
    end
end
clear images;
disp('Data Read');

% DEFINE AN ARCHITECTURE
arch = struct('dataSize', dataSize, ...
              'nFM', 3, ...
              'filterSize', [7 7], ...
              'stride', [2 2], ...
              'inputType', 'binary');

% GLOBAL OPTIONS
arch.opts = {'nEpoch', 5, ...
             'lRate', .00001, ...
             'displayEvery',1, ...
             'sparsity', .02, ...
             'sparseGain', 5};% , ...
%            'visFun', @visBinaryCRBMLearning}; % UNCOMMENT TO VIEW LEARNING

% INITIALIZE AND TRAIN
cr = crbm(arch);
cr = cr.train(Data);
disp('CRBM Training done for data');

i = 1;
while ~isempty(Data)
    % INFER HIDDEN AND POOLING LAYER EXPECTATIONS
    % CONDITIONED ON SOME INPUT
    [cr,ep] = cr.poolGivVis(Data(:,:,1));
    cr = cr.hidGivVis(Data(:,:,1));
    [nCols,nRows,k]=size(cr.eHid);
    features(i,:) = reshape(cr.eHid,nRows*nCols*k,1);
    i = i + 1;
    Data(:,:,1) = [];
end
clear Data;
clear cr;
save('features.mat','features','-v7.3');
disp('Features extracted');
%Features = load('features.mat');
%features = Features.features;
    
numTestRun = length(expToImg);
finalResult = cell(numTestRun,1);
finalResult(:) = {cell(length(tags),2)};
for testRun = 1 : numTestRun;
    trainTag = cell(1,1);
    testTag = cell(1,1);
    trainImgCount = 1;
    testImgCount = 1;
    for i = 1 : numTestRun
        offset = 0;
        if i - 1 >= 1
            for k = 1 : (i - 1)
                offset = offset + expToImg{k,1};
            end
        end
        if i ~= testRun
            for j = 1 : expToImg{i,1}
                trainFeatures(trainImgCount,:) = features(j+offset,:);
                trainTag{trainImgCount,1} = tag{j+offset,1};
                trainImgCount = trainImgCount + 1;
            end
        else
            for j = 1 : expToImg{i,1}
                testFeatures(testImgCount,:) = features(j+offset,:);
                testTag{testImgCount,1} = tag{j+offset,1};
                testImgCount = testImgCount + 1;
            end
        end
    end
    disp('Training and Test data created');
    clear images;


    models = cell(length(tags),1);
    % SVM Training
    for i = 1 : length(tags)
        trainLabel = ones((trainImgCount - 1), 1)*-1;
        for j = 1 : (trainImgCount - 1)
            if ismember(tags{i,1}, trainTag{j,1})
                trainLabel(j,1) = 1;
            end
        end
        models{i,1} = svmtrain(double(trainLabel),double(trainFeatures));
    end
    clear trainFeatures;
    clear trainLabel;

    disp('SVM traininig done');
    % Predict
    for i = 1 : length(tags)
        testLabel = ones((testImgCount - 1), 1)*-1;
        for j = 1 : (testImgCount - 1)
            if ismember(tags{i,1}, testTag{j,1})
                testLabel(j,1) = 1;
            end
        end
        [predict_label, accuracy, dec_values] = svmpredict(double(testLabel),double(testFeatures),models{i,1});
        finalResult{testRun}{i,1} = [predict_label, testLabel, tags{i,1}*ones((testImgCount - 1), 1)];
        finalResult{testRun}{i,2} = accuracy;
    end
    disp('Prediction done');
    
    clear testFeatures;
    clear testLabel;
    clear models;
end

clear features;
