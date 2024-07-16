clc
clear
close all

path = 'E:\ADNI\connectivity_bold_lmci\';
dir_path = dir(path);

% VCI = load("E:\ly\data\MMD_graph\MMD_334_BOLD_FNC.mat");
% mean_rois = VCI.Node_feature;
% [~,~,subnum] = size(mean_rois);

%% 动态滑窗
% win_len = 105;
win_lens = [70,75,80,85,90,95,100,105,110];
step = 1;
for win = 1:length(win_lens)
    win_len = win_lens(win);
    for sub = 3:length(dir_path)
        sub
        
        load([path, dir_path(sub).name]);
    
        roi = mean_roi;
        timepoints = size(roi, 1);
        win_num = floor((timepoints - win_len)/step) + 1;
        
        DFNC = [];
        DFNC_binary = [];
        for win = 1:win_num
            start = 1 + step*(win - 1);
            endd = win_len + step*(win - 1);
            tmp = roi(start:endd, :);
            net_tmp = corr(tmp);
            net_tmp = net_tmp - diag(diag(net_tmp));
    
            net_tmp = abs(net_tmp);
            net_tmp(net_tmp < 0.2) = 0;
            net_tmp_binary = net_tmp;
            net_tmp_binary(net_tmp_binary > 0) = 1;
    
            DFNC = cat(3, DFNC, net_tmp);
            DFNC_binary = cat(3, DFNC_binary, net_tmp_binary);
        end
        if ~exist('E:\ADNI\connectivity_bold_lmci_dfnc\')
               mkdir('E:\ADNI\connectivity_bold_lmci_dfnc\')
        end
        tar_path = ['E:\ADNI\connectivity_bold_lmci_dfnc\', 'win_',num2str(win_len),'_Sub', num2str(sub-2, '%03d'),'_DFNC_90.mat'];
        save(tar_path,'DFNC_binary','mean_roi','-v7.3');
    end
end

