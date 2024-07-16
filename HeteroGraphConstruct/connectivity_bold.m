% static FC

clc
clear
close all

path = 'E:\ADNI\MCI\BOLD\FunImgARFWS\';

dir_path = dir(path);
AAL = load_untouch_nii('AAL_61x73x61_YCG.nii');
AAL_img = AAL.img;
FNC_all = [];

window = 3;
step = 3;
timepoints = 690;
len = floor(((timepoints - window)/ step)) + 1;

for i = 1:length(dir_path)
    i
    cur_path = [path, dir_path(i).name,'\swFiltered_4DVolume.nii'];
    if exist(cur_path)
       subject = load_untouch_nii(cur_path);
       img = subject.img;
       timepoint = size(img,4);
       mean_roi = [];
       for j = 1:90
           tmp = AAL_img;
           tmp(find(tmp~=j)) = 0;
           tmp(find(tmp>0)) = 1;  %第j个脑区是1,其余都是0,做成mask
           temp = [];
           for k = 1:timepoint
               roi_per_timepoint = img(:,:,:,k).* single(tmp);   %获取该病人的第j个脑区在第k个时间点的区域
               mean_per_timepoint = mean(mean(mean(roi_per_timepoint))); 
               temp = [temp;mean_per_timepoint];  
           end
           mean_roi = [mean_roi,temp];
       end
       FNC = corr(mean_roi);  
       FNC = FNC - diag(diag(FNC));    % remove selfconnection (set to zeros)
       FNC_all = cat(3,FNC_all,FNC);
       
       if ~exist('E:\ADNI\connectivity_bold_lmci\')
           mkdir('E:\ADNI\connectivity_bold_lmci\')
       end
       tar_path = ['E:\ADNI\connectivity_bold_lmci\',dir_path(i).name,'_static_FNC_90.mat'];
       save(tar_path,'FNC','mean_roi','-v7.3');
    else
       display(dir_path(i).name);
    end
end
