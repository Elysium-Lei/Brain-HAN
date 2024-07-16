
clc
clear
close all

path = 'E:\ADNI\connectivity_bold_lmci_dfnc\';
dir_path = dir(path);

%% DFNC转FNC
m1 = [0 1 0;1 0 0;0 0 0];
m2 = [0 1 1;1 0 0;1 0 0];
m3 = [0 1 0;1 0 1;0 1 0];
m4 = [0 1 1;1 0 1;1 1 0];
FNC_all = [];

for sub = 3:length(dir_path)
    sub
    
    FNC = zeros(90,90);
    load([path, dir_path(sub).name]);
    dfnc = DFNC_binary;
    
    win_num = size(dfnc, 3);
    for win = 1:win_num
        fnc = dfnc(:,:,win);
        for i = 1:90
            for j = i+1:90
                for k = j+1:90
                    subgraph = fnc([i j k], [i j k]);
                    if isequal(subgraph, m1)
                        FNC(i,j) = FNC(i,j) + 0.01;
                    end
                    if isequal(subgraph, m2)
                        FNC(i,j) = FNC(i,j) + 0.02;
                        FNC(i,k) = FNC(i,k) + 0.02;
                    end
                    if isequal(subgraph, m3)
                        FNC(i,j) = FNC(i,j) + 0.02;
                        FNC(j,k) = FNC(j,k) + 0.02;
                    end
                    if isequal(subgraph, m4)
                        FNC(i,j) = FNC(i,j) + 0.1;
                        FNC(j,k) = FNC(j,k) + 0.1;
                        FNC(i,k) = FNC(i,k) + 0.1;
                    end
                end
            end
        end
    end
    FNC = (FNC - min(min(FNC)))/(max(max(FNC)) - min(min(FNC)));
    FNC(FNC < 0.4) = 0;
%     FNC(FNC > 0) = 1;
    FNC = FNC + FNC';
    FNC_all = cat(3, FNC_all, FNC);
end

load("E:\ADNI\graph\LMCI_BOLD_FNC.mat")
Graph_all = FNC_all;
Graph_all_binary = FNC_all;
Label_all = [Label_all;Label_all;Label_all;Label_all;Label_all;Label_all;Label_all;Label_all;Label_all];
Node_feature = cat(3, Node_feature,Node_feature,Node_feature,Node_feature,Node_feature,Node_feature, Node_feature, Node_feature, Node_feature);
save("E:\ADNI\graph\LMCI_BOLD_DFNC_210.mat", "Graph_all", "Graph_all_binary", "Label_all", "Node_feature")
    