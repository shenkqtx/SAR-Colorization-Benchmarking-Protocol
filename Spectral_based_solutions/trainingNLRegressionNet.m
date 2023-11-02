clc;
close all;
clear all;

%%% Loading Dataset
listDatasets;
GT = zeros(256,256,3,numel(listGT));
SAR = zeros(256,256,numel(listGT));
for ii = 1 : numel(listGT)
    GT(:,:,:,ii) = imread(listGT{ii});
    SAR_c = imread(listSAR{ii});
    %% preprocess
    SAR_c = SAR_c(:,:,1);
    SAR_c = SAR_c - min(SAR_c(:));
    SAR_c = SAR_c/max(SAR_c(:))*2^12;
    SAR(:,:,ii) = SAR_c;
end

SAR = double(SAR);
GT = double(GT);
size(SAR(:)');

%% Training NL Regression Network
net = fitnet([1,3],'trainlm');
band1 = GT(:,:,1,:);
band2 = GT(:,:,2,:);
band3 = GT(:,:,3,:);

net = train(net,SAR(:)'./2^12,[band1(:)';band2(:)';band3(:)']./2^12); 

%% View network
view(net)

%% Save model
save modelNLRegressionNet.mat net