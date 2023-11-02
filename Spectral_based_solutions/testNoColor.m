clc;
clear all;
close all;

%% Load Test Set
listDatasets;  
test_num = numel(listGTVal);
GT = cell(1,test_num);
SAR = cell(1,test_num);
SARcol = cell(1,test_num);
output_path = strcat('./NoColorOutput/');
if exist(output_path, 'dir')==0
    mkdir(output_path);
end

for i = 1:test_num
    GT{i} = double(imread(listGTVal{i})); 
    SAR{i} = double(imread(listSARVal{i}));
    SAR{i} = SAR{i}(:,:,1);
    
    %%% Stretch (preprocess) and duplication
    SAR{i} = SAR{i} - min(SAR{i}(:));
    SAR{i} = SAR{i}./max(SAR{i}(:))*2^12; % 12 is the bit depth of Sentinel-2 images
    SARcol{i} = repmat(SAR{i},[1 1 3]); 
    
    %%% View results
    %figure, imshow([uint8(GT{i}./2^12*255),uint8(SARcol{i}./2^12*255)])  % raw bits image show
    
    %%% Save Colorized images
    name = strsplit(listGTVal{i},'/');
    name = strsplit(name{end},'_');
    filename = strcat(output_path,name(1),'_',name(2),'_sarcolor_',name(4),'_',name(5));
    imwrite(uint16(SARcol{i}), filename{1})
end