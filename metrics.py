# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:59:37 2023

@author: 雷雨
"""

import torch
import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, auc, roc_curve

def accuracy(logits, labels):
    """Accuracy, auc with masking.Acc of the masked samples"""
    correct_prediction = np.equal(np.argmax(logits, 1), labels).astype(np.float32)
    return np.sum(correct_prediction), np.mean(correct_prediction)

def roc_auc(logits, labels):
    ''' input: logits, labels  ''' 
    preds = np.argmax(logits, 1)
    con_mat = confusion_matrix(labels, preds)
    if con_mat.shape[0] == 1:
        sen, spe = 1, 1
    else:      
        tp, fp, tn, fn = con_mat[1,1], con_mat[0,1], con_mat[0,0], con_mat[1,0]
        if tp+fn == 0:
            sen = 1
        else:
            sen = tp/(tp+fn)
        
        if fp+tn == 0:
            spe = 1
        else:
            spe = tn/(fp+tn)
    
    pos_probs = softmax(logits,1)[:,1]
    fpr, tpr, _ = roc_curve(labels, pos_probs, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    
    return roc_auc, sen, spe

def ppv_npv(output, target):

    # 取得到分类分数最大的值，返回第一维度是value，第二维度是index
    _, pred = output.max(1) 
    # 将 pred 展开成 one-hot编码形式
    pre_mask = torch.zeros(output.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
    # 将 target 也展开成 one-hot编码形式
    tar_mask = torch.zeros(output.size()).scatter_(1, target.data.cpu().view(-1, 1), 1.)
    # 计算 acc 的one-hot编码形式
    acc_mask = pre_mask * tar_mask
    
    temp = acc_mask.sum(0) / pre_mask.sum(0)   # 第一列是为负样本的个数，第二列是正样本的个数
    
    # 转换成numpy()
    ppv = temp[1].numpy()
    npv = temp[0].numpy()

    return ppv,npv