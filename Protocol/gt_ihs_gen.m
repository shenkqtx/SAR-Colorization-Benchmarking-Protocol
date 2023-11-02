%% Protocol of creating ground truth for SAR coloirizaiton
clc;
clear all;
close all;

%% load Set
sar_path = {'..\SEN12MS_CR_SARColorData\sar_train\ROIs1158_spring_s1_1_p169.tif','..\SEN12MS_CR_SARColorData\sar_train\ROIs1158_spring_s1_17_p825.tif'}; % add all the file paths of the sar images to this cell.
opt_path = {'..\SEN12MS_CR_SARColorData\opt_train\ROIs1158_spring_s2_1_p169.tif','..\SEN12MS_CR_SARColorData\opt_train\ROIs1158_spring_s2_17_p825.tif'}; % add all the file paths of the optical images to this cell.

num = size(sar_path,2);

output_path = strcat('..\SEN12MS_CR_SARColorData\gt_train\');
if exist(output_path, 'dir')==0
    mkdir(output_path);
end

for i = 1:num
    SAR1 = double(imread(sar_path{i}));
    SAR1 = SAR1(:,:,1); % choose the first band of Sentinel-1 image
    OPT = double(imread(opt_path{i}));
    OPTRGB = zeros(256,256,3);
    OPTRGB(:,:,1) = OPT(:,:,4); % Red band of Sentinel-2 image
    OPTRGB(:,:,2) = OPT(:,:,3); % Green band of Sentinel-2 image
    OPTRGB(:,:,3) = OPT(:,:,2); % Blue band of Sentinel-2 image
    
    %%% IHS fusion
    I = mean(OPTRGB,3);
    imageHR = (SAR1 - mean2(SAR1)).*(std2(I)./std2(SAR1)) + mean2(I);
    D = imageHR - I;
    GT = OPTRGB + repmat(D,[1 1 size(OPTRGB,3)]);
    
    %%% View results
    % figure, imshow([uint8(OPTRGB{i}./2^12*255), uint8(GT{i}./2^12*255)])
    
    %%% Save GT images
    name = strsplit(sar_path{i},'/');
    name = strsplit(name{end},'_');
    filename = strcat(output_path,name(1),'_',name(2),'_gt_',name(4),'_',name(5));
    imwrite(uint16(GT), filename{1})

end