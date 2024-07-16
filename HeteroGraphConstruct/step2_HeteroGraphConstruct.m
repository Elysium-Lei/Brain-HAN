clc
clear
close all

bold = load("E:\ADNI\graph\ADNI_LMCI_BOLD_DFNC_210.mat");
dti = load("E:\ADNI\graph\ADNI_LMCI_DTI_FN_DFNC_210.mat");
outpath = 'E:\ADNI\graph\';

modality = [0 1 1;1 0 1;1 1 0];
HeteroGraph = [];
for sub = 1:length(bold.Label_all)
    sub

    A_bold_binary = bold.Graph_all_binary(:,:,sub);
    A_dti_binary = dti.Graph_all_binary(:,:,sub);
    %% 对BOLD、DTI模态下每个roi的连接模式计算余弦相似度
    A_fusion = [];
    for i = 1:90
        roi_bold = A_bold_binary(i,:);
        similarity = (roi_bold*A_dti_binary) ./ (sqrt(sum(roi_bold)) * sqrt(sum(A_dti_binary)));
        similarity(isnan(similarity)) = 0;
        [similarity, ind] = sort(similarity, 'descend');

        tmp = zeros(1, 90);
        tmp(ind(1:8)) = similarity(1:8);
        A_fusion = [A_fusion;tmp];
    end
    A_fusion = A_fusion - diag(diag(A_fusion));
    
    %% 将BOLD、DTI模态下同时出现的模体加入fusion邻接矩阵
    M = zeros(90, 90, 90);
    A_modality = zeros(90, 90);
    for i = 1:90
        for j = i+1:90
            for k = j+1:90
                subgraph_bold = A_bold_binary([i j k], [i j k]);
                subgraph_dti = A_dti_binary([i j k], [i j k]);
                if isequal(subgraph_bold, modality) && isequal(subgraph_dti, modality)
                    M(i, j, k) = 1;
                end
            end
        end
    end

    ind = find(M == 1);
    [x, y, z] = ind2sub(size(M), ind);
    for idx = 1:length(x)
        A_modality([x(idx), y(idx), z(idx)], [x(idx), y(idx), z(idx)]) = modality;
    end

    A_fusion = A_fusion + A_modality;

    
    A_bold = bold.Graph_all(:,:,sub);
    A_dti = dti.Graph_all(:,:,sub);
    HeteroGraph_sub = [A_bold A_fusion; A_fusion' A_dti];
    HeteroGraph = cat(3, HeteroGraph, HeteroGraph_sub);
end

HeteroGraph_binary = HeteroGraph;
HeteroGraph_binary(HeteroGraph > 0) = 1;
bold_node_feature = bold.Node_feature;
dti_node_feature = dti.Node_feature;
label = bold.Label_all;
save([outpath, 'HeteroGraph_ADNI_LMCI_DFNC_450.mat'], 'HeteroGraph', 'HeteroGraph_binary', 'bold_node_feature', 'dti_node_feature', 'label')