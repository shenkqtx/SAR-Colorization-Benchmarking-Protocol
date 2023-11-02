%% data analysis about the linear relationship between the SAR band and ground truth bands to support the reasonability of spectral-based solutions.
clc;
close all;
clear all;

test = 2;
switch test
    case 1
        GTname = 'ROIs1158_spring_gt_1_p169.tif';
        SARname = 'ROIs1158_spring_s1_1_p169.tif';
    case 2
        GTname = 'ROIs1158_spring_gt_44_p41.tif';
        SARname = 'ROIs1158_spring_s1_44_p41.tif';
end


cd 'gt_test'
GT = imread(GTname);
cd ..
cd 'sar_test'
SAR = imread(SARname);
cd ..

SAR = double(SAR);
SAR = SAR(:,:,1);
GT = double(GT);

x = [SAR(:),ones(numel(SAR(:)),1)];  

band1 = GT(:,:,1); 
[b1,~,~,~,stats] = regress(band1(:),x);
figure, plot(SAR(:),band1(:),'.k')
hold on, plot([-25:0], b1(1).*[-25:0]+b1(2),'-r','LineWidth',2)
Rsquared1 = stats(1)  

band2 = GT(:,:,2); 
[b2,~,~,~,stats] = regress(band2(:),x);
figure, plot(SAR(:),band2(:),'.k')
hold on, plot([-25:0], b2(1).*[-25:0]+b2(2),'-r','LineWidth',2)
Rsquared2 = stats(1)

band3 = GT(:,:,3); 
[b3,~,~,~,stats] = regress(band3(:),x);
figure, plot(SAR(:),band3(:),'.k')
hold on, plot([-25:0], b3(1).*[-25:0]+b3(2),'-r','LineWidth',2)
Rsquared3 = stats(1)