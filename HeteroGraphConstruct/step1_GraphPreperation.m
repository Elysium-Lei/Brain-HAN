% ������������GCN��graphԭ����
clc
clear
close all

boldpath = 'E:\ADNI\connectivity_bold_emci\';
outpath = 'E:\ADNI\graph\';
if ~exist(outpath)
    mkdir(outpath)
end

dir_bold = dir(boldpath);

Graph_all = [];
Node_feature = [];

% all_label = readtable('.\all_label.xlsx', 'Sheet', 1);
% Label_all = table2array(all_label(:,6));

% Label_all = zeros(length(dir_bold)-2, 1);
Label_all = ones(length(dir_bold)-2, 1);

%% �����ж����fMRI��FN��mean-roi����ƴ������
for i = 3:length(dir_bold)           
    load([boldpath,dir_bold(i).name]) 
    
    Graph_all = cat(3,Graph_all,FNC);

    fea = mean_roi';
    for j = 1:size(fea,2)
        fea(:,j)  = ((fea(:,j)-min(fea(:,j)))/(max(fea(:,j))-min(fea(:,j))))*2-1;
    end
    Node_feature = cat(3,Node_feature,fea);
end

%% ��FNC��ֵ����ֻ�����ϵ���ľ���ֵ���ڵ���0.2ʱ����Ϊ��أ�����Ϊ0
Graph_all = abs(Graph_all);

for i = 1:size(Graph_all,3)
    tmp = Graph_all(:,:,i);
    tmp(find(tmp<0.2)) = 0;
    Graph_all(:,:,i) = tmp;
end

Graph_all_binary = Graph_all;
Graph_all_binary(find(Graph_all>0)) = 1;

save([outpath, 'EMCI_BOLD_FNC.mat'], "Graph_all", "Graph_all_binary", "Label_all", "Node_feature");

%% �����ж����DTI��FN��grey featureƴ������
clearvars -except Label_all outpath

dtipath = 'E:\ADNI\connectivity_dti_emci\';
dir_dti = dir(dtipath);

Graph_all = [];
Node_feature = [];

for i = 3:length(dir_dti)
    load([dtipath, dir_dti(i).name]);
    
    Graph_all = cat(3, Graph_all, FN);

    dti_fea = fea';
    for j = 1:size(dti_fea, 2)
        dti_fea(:,j) = ((dti_fea(:,j) - min(dti_fea(:,j)))/(max(dti_fea(:,j)) - min(dti_fea(:,j))))*2 - 1;
    end
    Node_feature = cat(3, Node_feature, dti_fea);
end


%% ����������dti����ֵ������ά������������ڵ���5������Ч���ӣ����ҽ����˹淶��
Graph_all(find(Graph_all<5)) = 0;
for i = 1:size(Graph_all,3)
    tmp = Graph_all(:,:,i);
    mmax = max(max(tmp));
    mmin = min(min(tmp(tmp~=0)));
    tmp = (tmp - mmin)/(mmax-mmin) *2;
    tmp(find(tmp<0)) = 0;
    Graph_all(:,:,i) = tmp;
end

Graph_all_binary = Graph_all;
Graph_all_binary(find(Graph_all>0)) = 1;

save([outpath, 'EMCI_DTI_FN.mat'], "Graph_all", "Graph_all_binary", "Label_all", "Node_feature");

%% ���´��뽫��ͬ�����ݵ�ͼƴ����һ��
outpath = 'E:\ADNI\graph\';

bold1 = load("E:\ADNI\graph\LMCI_BOLD_DFNC_210.mat");
bold2 = load("E:\ADNI\graph\LMCI_BOLD_FNC.mat");
bold3 = load("E:\ADNI\graph\NC_BOLD_FNC.mat");

Graph_all = cat(3, bold1.Graph_all, bold2.Graph_all, bold3.Graph_all);
Graph_all_binary = cat(3, bold1.Graph_all_binary, bold2.Graph_all_binary, bold3.Graph_all_binary);
Label_all = cat(1, bold1.Label_all, bold2.Label_all, bold3.Label_all);
Node_feature = cat(3, bold1.Node_feature, bold2.Node_feature, bold3.Node_feature);

save([outpath, 'ADNI_LMCI_BOLD_DFNC_210.mat'], "Graph_all", "Graph_all_binary", "Label_all", "Node_feature");

clearvars -except outpath

dti1 = load("E:\ADNI\graph\LMCI_DTI_FN.mat");
dti2 = load("E:\ADNI\graph\NC_DTI_FN.mat");

Graph_all = cat(3, dti1.Graph_all,dti1.Graph_all,dti1.Graph_all,dti1.Graph_all,dti1.Graph_all,dti1.Graph_all,dti1.Graph_all, dti1.Graph_all, dti1.Graph_all, dti1.Graph_all, dti2.Graph_all);
Graph_all_binary = cat(3, dti1.Graph_all_binary,dti1.Graph_all_binary,dti1.Graph_all_binary,dti1.Graph_all_binary,dti1.Graph_all_binary,dti1.Graph_all_binary,dti1.Graph_all_binary, dti1.Graph_all_binary, dti1.Graph_all_binary, dti1.Graph_all_binary, dti2.Graph_all_binary);
Label_all = cat(1, dti1.Label_all,dti1.Label_all,dti1.Label_all,dti1.Label_all,dti1.Label_all,dti1.Label_all,dti1.Label_all, dti1.Label_all, dti1.Label_all, dti1.Label_all, dti2.Label_all);
Node_feature = cat(3, dti1.Node_feature,dti1.Node_feature,dti1.Node_feature,dti1.Node_feature,dti1.Node_feature,dti1.Node_feature,dti1.Node_feature, dti1.Node_feature,dti1.Node_feature,dti1.Node_feature, dti2.Node_feature);

save([outpath, 'ADNI_LMCI_DTI_FN_DFNC_210.mat'], "Graph_all", "Graph_all_binary", "Label_all", "Node_feature");
