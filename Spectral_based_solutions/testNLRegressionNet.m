clc;
clear all;
close all;

%% Load test image
listDatasets;
test_num = numel(listGTVal);
GT = cell(1,test_num);
SAR = cell(1,test_num);
SARcol = cell(1,test_num);
output_path = strcat('./NLRegressionOutput/');
if exist(output_path, 'dir')==0
    mkdir(output_path);
end

%% Load model
load modelNLRegressionNet.mat

for i = 1:test_num
    GT{i} = double(imread(listGTVal{i}));
    SAR{i} = double(imread(listSARVal{i}));
    SAR{i} = SAR{i}(:,:,1);
    
    %% preprocess
    SAR{i} = SAR{i} - min(SAR{i}(:));
    SAR{i} = SAR{i}./max(SAR{i}(:))*2^12;
    
    %% Colorization
     SARcol{i} = reshape(net(SAR{i}(:)'./(2^12))'*(2^12),[256,256,3]);
    
    %% View results
%     figure, imshow([uint8(GT{i}./2^12*255),uint8(SARcol{i}./2^12*255)])  % raw bits image show
    
    %% Save Colorized images
    name = strsplit(listGTVal{i},'/');
    name = strsplit(name{end},'_');
    filename = strcat(output_path,name(1),'_',name(2),'_sarcolor_',name(4),'_',name(5));
    imwrite(uint16(SARcol{i}), filename{1})
end