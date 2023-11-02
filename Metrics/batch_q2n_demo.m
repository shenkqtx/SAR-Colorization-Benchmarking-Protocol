clear; close all; clc;

method = {'NoColor','LinearRegression','NLRegression',};
method_size = size(method);
num = method_size(2);  

%% Load GT Testing Set
listTestingDatasets;
test_num = numel(listGTVal); 
 
index_path = 'sarcolor_q2n_index.xls';
xlswrite(index_path,[{char('method')},{char('Q2n')}],'sheet1','A1:B1');

for j = 1:num
    GT = cell(1,test_num);
    SARcol = cell(1,test_num);
    Q2n_list = [];
    
    for i = 1:test_num
        GT{i} = double(imread(listGTVal{i}));
        name = strsplit(listGTVal{i},'/');  
        name = strsplit(name{end},'_');  
        
        output_path = strcat('../', char(method(j)), 'Output/');
        sarcolor_filename = strcat(output_path,name(1),'_',name(2),'_sarcolor_',name(4),'_',name(5));
        SARcol{i} = double(imread(char(sarcolor_filename(1))));

        GT_4bands = zeros(256,256,4);
        SARcol_4bands = zeros(256,256,4);
        zero_band = zeros(256,256);
        
        GT_4bands(:,:,1:3) = GT{i};
        GT_4bands(:,:,4) = zero_band;
        SARcol_4bands(:,:,1:3) = SARcol{i};
        SARcol_4bands(:,:,4) = zero_band;
        
        Q2n_list(i) = q2n(GT_4bands,SARcol_4bands, 32, 16);

    end
    index = strcat(num2str(roundn(mean(Q2n_list),-4)),'¡À',num2str(roundn(std(Q2n_list),-4)))
    xlswrite(index_path,[{char(method(j))},{strcat(num2str(roundn(mean(Q2n_list),-4)),'¡À',num2str(roundn(std(Q2n_list),-4)))}],'sheet1',['A',num2str(j+1)]);
end




