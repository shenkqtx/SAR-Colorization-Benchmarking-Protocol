clc;
close all;
clear all;

%% Load Dataset
listDatasets;
GT = zeros(256,256,3,numel(listGT));
SAR = zeros(256,256,numel(listGT));
for ii = 1 : numel(listGT)
    GT(:,:,:,ii) = imread(listGT{ii});
    SAR_c = imread(listSAR{ii});
    
    %% preprocess for the SAR images
    SAR_c = SAR_c(:,:,1);
    SAR_c = SAR_c - min(SAR_c(:));
    SAR_c = SAR_c./max(SAR_c(:))*2^12;
    SAR(:,:,ii) = SAR_c;
end

SAR = double(SAR);
GT = double(GT);

%% Add bias estimation
x = [SAR(:),ones(numel(SAR(:)),1)]; 

%% Regression for the three bands
band1 = reshape(GT(:,:,1,:),[numel(GT(:,:,1,:)),1]);
b1 = regress(band1(:),x); 

band2 = reshape(GT(:,:,2,:),[numel(GT(:,:,2,:)),1]); 
b2 = regress(band2(:),x);

band3 = reshape(GT(:,:,3,:),[numel(GT(:,:,3,:)),1]); 
b3 = regress(band3(:),x);

%% Save model
save 'modelLinearRegression.mat' b1 b2 b3 

%% Load model
load modelLinearRegression.mat

%% Load Test Set
test_num = numel(listGTVal);
GT = cell(1,test_num);
SAR = cell(1,test_num);
SARcol = cell(1,test_num);
output_path = strcat('./LinearRegressionOutput/bias/');

if exist(output_path, 'dir')==0
    mkdir(output_path);
end

for i = 1:test_num
    GT{i} = double(imread(listGTVal{i}));
    SAR{i} = double(imread(listSARVal{i}));
    SAR{i} = SAR{i}(:,:,1);
    
    %% preprocess
    SAR{i} = SAR{i} - min(SAR{i}(:));
    SAR{i} = SAR{i}./max(SAR{i}(:))*2^12;
    
    %% Colorization with bias
    SARcol{i}(:,:,1) = b1(1).*SAR{i} + b1(2);
    SARcol{i}(:,:,2) = b2(1).*SAR{i} + b2(2);
    SARcol{i}(:,:,3) = b3(1).*SAR{i} + b3(2);

    %% Colorization without bias
%     SARcol{i}(:,:,1) = b1(1).*SAR{i};
%     SARcol{i}(:,:,2) = b2(1).*SAR{i};
%     SARcol{i}(:,:,3) = b3(1).*SAR{i};
    
    %% View results
%     figure, imshow([uint8(GT{i}./2^12*255),uint8(SARcol{i}./2^12*255)])  % raw bits image show

    %% Save Colorized images
    name = strsplit(listGTVal{i},'/');
    name = strsplit(name{end},'_');
    filename = strcat(output_path,name(1),'_',name(2),'_sarcolor_',name(4),'_',name(5));
    imwrite(uint16(SARcol{i}), filename{1})
end