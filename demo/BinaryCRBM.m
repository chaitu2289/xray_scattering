%dataset = 'mnistSmall';

%[images, tags] load('data.mat');
numTestRun = 1;
finalResult = cell(numTestRun);
for testRun = 1 : numTestRun;
    [images, tags, m, n] = loadImagesTags;
    disp('Data Read');
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
    disp('Training and Test data created');
    clear images;

    % DEFINE AN ARCHITECTURE
    arch = struct('dataSize', dataSize, ...
        	'nFM', 3, ...
            'filterSize', [7 7], ...
            'stride', [2 2], ...
            'inputType', 'binary');

    % GLOBAL OPTIONS
    arch.opts = {'nEpoch', 1, ...
        		 'lRate', .00001, ...
            	 'displayEvery',1, ...
                 'sparsity', .02, ...
                'sparseGain', 5};% , ...
%               'visFun', @visBinaryCRBMLearning}; % UNCOMMENT TO VIEW LEARNING

    % INITIALIZE AND TRAIN
    cr = crbm(arch);
    cr = cr.train(trainData);
    disp('CRBM Training done for trainData');

    i = 1;
    while ~isempty(trainData)
        % INFER HIDDEN AND POOLING LAYER EXPECTATIONS
        % CONDITIONED ON SOME INPUT
        [cr,ep] = cr.poolGivVis(trainData(:,:,1));
        cr = cr.hidGivVis(trainData(:,:,1));
        [nCols,nRows,k]=size(cr.eHid);
        trainFeatures(i,:) = reshape(cr.eHid,nRows*nCols*k,1);
        i = i + 1;
        trainData(:,:,1) = [];
    end
    clear trainData;
    clear cr;
    disp('Train features extracted');

    % INITIALIZE AND TRAIN
    cr = crbm(arch);
    cr = cr.train(testData);
    disp('CRBM Training done for testData');
    
    i = 1;
    while ~isempty(testData)
        % INFER HIDDEN AND POOLING LAYER EXPECTATIONS
        % CONDITIONED ON SOME INPUT
        [cr,ep] = cr.poolGivVis(testData(:,:,1));
        cr = cr.hidGivVis(testData(:,:,1));
        [nCols,nRows,k]=size(cr.eHid);
        testFeatures(i,:) = reshape(cr.eHid,nRows*nCols*k,1);
        i = i + 1;
        testData(:,:,1) = [];
    end
    clear testData;
    clear cr;
    disp('Test features extracted');

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
    %result = cell(length(tags),2);
    for i = 1 : length(tags)
        testLabel = ones((testImgCount - 1), 1)*-1;
        for j = 1 : (testImgCount - 1)
            if ismember(tags{i,1}, testTag{j,1})
                testLabel(j,1) = 1;
            end
        end
        [predict_label, accuracy, dec_values] = svmpredict(double(testLabel),double(testFeatures),models{i,1});
        finalResult{testRun,i,1} = [predict_label, testLabel, tags{i,1}*ones((testImgCount - 1), 1)];
        finalResult{testRun,i,2} = accuracy;
    end
    disp('Prediction done');
    
    clear testFeatures;
    clear testLabel;
    clear models;
end