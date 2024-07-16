clc
clear
close all

fn_path = 'E:\ADNI\LMCI\DTI\output_panda\';
fea_path = 'E:\ADNI\DTI_fea_lmci\';
outpath = 'E:\ADNI\connectivity_dti_lmci\';
if ~exist(outpath)
    mkdir(outpath)
end

dir_fn = dir(fn_path);
dir_fea = dir(fea_path);

for i=3:length(dir_fea)
    i
    FN = load([fn_path,dir_fn(i).name,'\Network\Deterministic\',dir_fn(i).name,'_dti_FACT_45_02_1_0_Matrix_FN_AAL_Contract_90_2MM_90.txt']);
    FN = single(FN);
    Fea = load([fea_path,dir_fea(i).name,'\GrayFeature_ROI_DTI_90.mat']);
    fea = Fea.fea;

    save([outpath,dir_fea(i).name,'_dti_fn_90.mat'], 'FN', 'fea', '-v7.3');
end