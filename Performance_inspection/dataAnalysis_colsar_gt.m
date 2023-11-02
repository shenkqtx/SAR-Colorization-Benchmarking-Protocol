clc;
close all;
clear all;

test = 4;
cd 'dataAnalysis_colsar_gt'
method = 'can4colsar' % nocolsar, lr4colsar, nl4colsar, cnn4colsar
switch test
    case 1
        GTname = 'ROIs1158_spring_gt_8_p251.tif';
        ColSARname = strcat('ROIs1158_spring_', method, '_8_p251.tif');
    case 2
        GTname = 'ROIs1158_spring_gt_119_p495.tif';
        ColSARname = strcat('ROIs1158_spring_',method,'_119_p495.tif');
    case 3
        GTname = 'ROIs1158_spring_gt_120_p408.tif';
        ColSARname = strcat('ROIs1158_spring_', method, '_120_p408.tif');
    case 4
        GTname = 'ROIs1158_spring_gt_121_p560.tif';
        ColSARname = strcat('ROIs1158_spring_', method, '_121_p560.tif');
end

GT = double(imread(GTname));
ColSAR = double(imread(ColSARname));

gt_band1 = GT(:,:,1); 
colsar_band1 = ColSAR(:,:,1); 

x = [gt_band1(:),ones(numel(gt_band1(:)),1)];

[b1,~,~,~,stats] = regress(colsar_band1(:),x);

% test=1, 1500£» test=2£¬2500 ; test=3, 2600£»test=4£¬2600
figure, plot(gt_band1(:),colsar_band1(:),'.k')
hold on, plot([0:2600], b1(1).*[0:2600]+b1(2),'-r','LineWidth',2)
hold on, plot([0:2600], [0:2600],'--b','LineWidth',2)
Rsquared1 = stats(1)  
saveas(gcf, strcat('scatterplots_',method, '.png'))


