function [ trainingData,m,n ] = loadImages()
    sourceDirectory = '/home/chaitu/computer_vision/project/medal/sample_image/';
    Files=dir(sourceDirectory);
    i = 1;
    trainingData = [];
    %for k=1:length(Files)
    for k = 1:3
        FileName=Files(k).name;
        if ~strcmp(FileName,'.') && ~strcmp(FileName,'..');
            %display(FileName);
            image =  imread(strcat(sourceDirectory,FileName));
            [m,n] = size(image);
            %trainingData(i,:) = reshape(image,[1,m*n]);
            trainingData = cat(3, trainingData, image);
            i = i + 1;
        end
    end
    trainingData = im2double(trainingData);
end