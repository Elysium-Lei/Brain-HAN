# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:11:02 2023

@author: Dell
"""

import os
import argparse
import logging
from tqdm import tqdm
import numpy as np
import scipy.io as sio
from pandas import DataFrame

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from HeteroDataLoad import HeteroDataPre, MyHeteroData
from utils import CosineScheduler, Kfold_random_split
from MMHAN_HGP import MultiHAN
from losses import FocalLoss
from metrics import accuracy, roc_auc

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./HeteroData', type=str, help='path of dataset')
parser.add_argument('--data_name', default='HeteroGraph_ADNI_M_DFNC', type=str, help='name of dataset')
parser.add_argument('--mask_name', default='train_val_test_mask_DFNC_4', type=str, help='name of mask')
parser.add_argument('--gpu', default='0', type=str, help='GPU ID')
parser.add_argument('--seed', default=6, type=int, help='manual seed')
parser.add_argument('--kfold', default=5, type=int, help='num of folds')
parser.add_argument('--epochs', default=200, type=int, help='num of epochs')
parser.add_argument('--warmup_epochs', default=10, type=int, help='warmup epochs for scheduler')
parser.add_argument('--const_epochs', default=0, type=int, help='const epochs for schedular')
parser.add_argument('--hidden_channels', default=128, type=int, help='num of hidden channels')
parser.add_argument('--out_channels', default=2, type=int, help='num of classes')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--dropout_ratio', default=0.45, type=float, help='dropout ratio')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

logging.info('Prepare data...')
HeteroDataPre(args.data_path, args.data_name)
Dataset = MyHeteroData(args.data_path)
train_sets, val_sets, test_set = Kfold_random_split(Dataset, os.path.join(args.data_path, args.mask_name+'.mat'), args.kfold, args.seed)

def test(model, loader, mode, fold):
    model.eval()
    test_acc = 0.
    test_auc = 0.
    test_sen = 0.
    test_spe = 0.
    test_loss = 0.
    test_B1 = []
    test_B2 = []
    test_B3 = []
    test_D1 = []
    test_D2 = []
    test_D3 = []
    test_outs = []
    test_features = []
    test_labels = []
    
    for ind, data in enumerate(loader):
        data = data.to(device)
        test_out, test_feature, test_P_bold1, test_P_dti1, test_P_bold2, test_P_dti2, test_P_bold3, test_P_dti3 = model(data)  
        
        loss_tmp = criterion(test_out, data.y_dict['bold'])
        
        test_logits = test_out.detach().cpu().numpy()
        test_label = data.y_dict['bold'].detach().cpu().numpy()
        
        test_loss += loss_tmp
        
        test_B1.append(test_P_bold1.detach().cpu().numpy())
        test_B2.append(test_P_bold2.detach().cpu().numpy())
        test_B3.append(test_P_bold3.detach().cpu().numpy())
        test_D1.append(test_P_dti1.detach().cpu().numpy())
        test_D2.append(test_P_dti2.detach().cpu().numpy())
        test_D3.append(test_P_dti3.detach().cpu().numpy())
        
        test_outs.append(test_logits)
        test_features.append(test_feature.detach().cpu().numpy())
        test_labels.append(test_label)
    
    test_outs = np.concatenate(test_outs, axis=0)
    test_features = np.concatenate(test_features, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)    
    
    correct, test_acc = accuracy(test_outs, test_labels) 
    test_auc, test_sen, test_spe = roc_auc(test_outs, test_labels)
    test_loss = test_loss/(ind+1)
    
    test_B1 = np.concatenate(test_B1, axis=0)
    test_B2 = np.concatenate(test_B2, axis=0)
    test_B3 = np.concatenate(test_B3, axis=0)
    test_D1 = np.concatenate(test_D1, axis=0)
    test_D2 = np.concatenate(test_D2, axis=0)
    test_D3 = np.concatenate(test_D3, axis=0)
    
    test_B1 = test_B1[:,:int(test_B1.shape[1]/2),:]
    test_B2 = test_B2[:,:int(test_B2.shape[1]/2),:]
    test_B3 = test_B3[:,:int(test_B3.shape[1]/2),:]
    test_D1 = test_D1[:,int(test_D1.shape[1]/2):,:]
    test_D2 = test_D2[:,int(test_D2.shape[1]/2):,:]
    test_D3 = test_D3[:,int(test_D3.shape[1]/2):,:]
    
    test_B1_c = np.argmax(test_B1, axis=2) + 1
    test_B2_c = np.argmax(test_B2, axis=2) + 1
    test_B3_c = np.argmax(test_B3, axis=2) + 1
    test_D1_c = np.argmax(test_D1, axis=2) + 1
    test_D2_c = np.argmax(test_D2, axis=2) + 1
    test_D3_c = np.argmax(test_D3, axis=2) + 1
    
    test_outpath = './cross_val_results/fold'+str(fold+1)+'/'+mode+'_results'
    if not os.path.exists(test_outpath):
        os.makedirs(test_outpath)
    sio.savemat(test_outpath+'/epoch{}.mat'.format(epoch), {'acc':test_acc, 'auc':test_auc, 'sen':test_sen, 'spe':test_spe,\
                                                            'B1':test_B1, 'B2':test_B2, 'B3':test_B3, 'D1':test_D1, 'D2':test_D2, 'D3':test_D3,\
                                                            'B1_cluster':test_B1_c, 'B2_cluster':test_B2_c, 'B3_cluster':test_B3_c, 'D1_cluster':test_D1_c, 'D2_cluster':test_D2_c, 'D3_cluster':test_D3_c,\
                                                            'labels':test_labels, 'outs':test_outs, 'features':test_features})
    
    return test_acc, test_auc, test_sen, test_spe, test_loss

for k in range(args.kfold):
    logging.info('Cross Validation fold'+str(k+1))
    train_set = train_sets[k]
    val_set = val_sets[k]
    
    trainLoader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valLoader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    testLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True) 
    
    logging.info('Prepare model...')
    node_types = Dataset.get(0).node_types
    metadata = Dataset.get(0).metadata()

    in_channels = {}
    in_channels[node_types[0]] = -1
    in_channels[node_types[-1]] = -1

    model = MultiHAN(in_channels, args.hidden_channels, args.out_channels, metadata, args.dropout_ratio).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineScheduler(optimizer, param_name='lr', t_max=args.epochs, value_min=args.lr * 1e-2,
                                    warmup_t=args.warmup_epochs, const_t=args.const_epochs)
    wd_scheduler = CosineScheduler(optimizer, param_name='weight_decay', t_max=args.epochs)
    criterion = FocalLoss()
    
    max_acc = 0.6
    patience = 0
    
    train_losses = []
    val_losses = []
    test_losses = []
    
    train_acc_epochs = []
    train_auc_epochs = []
    train_sen_epochs = []
    train_spe_epochs = []
    
    val_acc_epochs = []
    val_auc_epochs = []
    val_sen_epochs = []
    val_spe_epochs = []
    
    test_acc_epochs = []
    test_auc_epochs = []
    test_sen_epochs = []
    test_spe_epochs = []
    
    logging.info('Train model...')
    for epoch in tqdm(range(args.epochs)):
        model.train()
        tra_loss = 0.0
        tra_acc = 0.0
        tra_auc = 0.0
        tra_sen = 0.0
        tra_spe = 0.0
        B1 = []
        B2 = []
        B3 = []
        D1 = []
        D2 = []
        D3 = []
        outs = []
        features = []
        labels = []
    
        
        for ind, data in enumerate(trainLoader):
            optimizer.zero_grad()
            data = data.to(device)
            
            out, feature, P_bold1, P_dti1, P_bold2, P_dti2, P_bold3, P_dti3 = model(data)
            loss = criterion(out, data.y_dict['bold'])

            loss.backward()
            optimizer.step()
            lr_scheduler.step(epoch + 1)
            wd_scheduler.step(epoch + 1)

            logits = out.detach().cpu().numpy()
            label = data.y_dict['bold'].detach().cpu().numpy()
            
            correct_train, acc_train = accuracy(logits, label) 
            auc_train, sen_train, spe_train = roc_auc(logits, label)
            
            tra_acc += acc_train
            tra_auc += auc_train
            tra_sen += sen_train
            tra_spe += spe_train
            tra_loss += loss
            
            B1.append(P_bold1.detach().cpu().numpy())
            B2.append(P_bold2.detach().cpu().numpy())
            B3.append(P_bold3.detach().cpu().numpy())
            D1.append(P_dti1.detach().cpu().numpy())
            D2.append(P_dti2.detach().cpu().numpy())
            D3.append(P_dti3.detach().cpu().numpy())
            
            outs.append(logits)
            features.append(feature.detach().cpu().numpy())
            labels.append(label)
            
        
        tra_acc = tra_acc/(ind+1)
        tra_auc = tra_auc/(ind+1)
        tra_sen = tra_sen/(ind+1)
        tra_spe = tra_spe/(ind+1)
        tra_loss = tra_loss / (ind+1)
        train_losses.append(tra_loss.detach().cpu().numpy())
        train_acc_epochs.append(tra_acc)
        train_auc_epochs.append(tra_auc)
        train_sen_epochs.append(tra_sen)
        train_spe_epochs.append(tra_spe)
        
        B1 = np.concatenate(B1, axis=0)
        B2 = np.concatenate(B2, axis=0)
        B3 = np.concatenate(B3, axis=0)
        D1 = np.concatenate(D1, axis=0)
        D2 = np.concatenate(D2, axis=0)
        D3 = np.concatenate(D3, axis=0)
        
        B1 = B1[:,:int(B1.shape[1]/2),:]
        B2 = B2[:,:int(B2.shape[1]/2),:]
        B3 = B3[:,:int(B3.shape[1]/2),:]
        D1 = D1[:,int(D1.shape[1]/2):,:]
        D2 = D2[:,int(D2.shape[1]/2):,:]
        D3 = D3[:,int(D3.shape[1]/2):,:]
        
        B1_c = np.argmax(B1, axis=2) + 1
        B2_c = np.argmax(B2, axis=2) + 1
        B3_c = np.argmax(B3, axis=2) + 1
        D1_c = np.argmax(D1, axis=2) + 1
        D2_c = np.argmax(D2, axis=2) + 1
        D3_c = np.argmax(D3, axis=2) + 1
        
        outs = np.concatenate(outs, axis=0)
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        val_acc, val_auc, val_sen, val_spe, val_loss = test(model,valLoader, 'val', k)
        test_acc, test_auc, test_sen, test_spe, test_loss = test(model, testLoader, 'test', k)
        
        val_losses.append(val_loss.detach().cpu().numpy())
        val_acc_epochs.append(val_acc)
        val_auc_epochs.append(val_auc)
        val_sen_epochs.append(val_sen)
        val_spe_epochs.append(val_spe)
        
        test_losses.append(test_loss.detach().cpu().numpy())
        test_acc_epochs.append(test_acc)
        test_auc_epochs.append(test_auc)
        test_sen_epochs.append(test_sen)
        test_spe_epochs.append(test_spe)
      
        print("\nTra loss:{:.5f}  acc:{:.4f}  auc:{:.4f}  sen:{:.4f}  spe:{:.4f} , Val loss:{:.5f}  acc:{:.4f}  auc:{:.4f}  sen:{:.4f}  spe:{:.4f}".format(tra_loss,
              tra_acc, tra_auc, tra_sen, tra_spe, val_loss, val_acc, val_auc, val_sen, val_spe))
        print("Test accuracy:{:.4f}\t auc:{:.4f}\t sen:{:.4f}\t spe:{:.4f}".format(test_acc, test_auc, test_sen, test_spe))
        
        model_path = './cross_val_model/fold'+str(k+1)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(model.state_dict(),model_path+"/epoch{}.pth".format(epoch))
        
        result_path = './cross_val_results/fold'+str(k+1)+'/train_results'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        sio.savemat(result_path+'/epoch{}.mat'.format(epoch), {'acc':tra_acc, 'auc':tra_auc, 'sen':tra_sen, 'spe':tra_spe,\
                                                                'B1':B1, 'B2':B2, 'B3':B3, 'D1':D1, 'D2':D2, 'D3':D3,\
                                                                'B1_cluster':B1_c, 'B2_cluster':B2_c, 'B3_cluster':B3_c, 'D1_cluster':D1_c, 'D2_cluster':D2_c, 'D3_cluster':D3_c,\
                                                                'labels':labels, 'outs':outs, 'features':features})
            
        if (val_acc >= max_acc) and (val_acc<=tra_acc) :
            torch.save(model.state_dict(),'latest.pth')
            print("Model saved at epoch{}".format(epoch))
    
            max_acc = val_acc
            patience = 0
        else:
            patience += 1
    
    train_metrics = DataFrame({'AUC':train_auc_epochs, 'ACC':train_acc_epochs, 'SEN':train_sen_epochs, 'SPE':train_spe_epochs})
    train_metrics.to_excel('./cross_val_results/fold'+str(k+1)+'/train_metrics.xlsx', sheet_name='sheet1', index=False)
    
    val_metrics = DataFrame({'AUC':val_auc_epochs, 'ACC':val_acc_epochs, 'SEN':val_sen_epochs, 'SPE':val_spe_epochs})
    val_metrics.to_excel('./cross_val_results/fold'+str(k+1)+'/val_metrics.xlsx', sheet_name='sheet1', index=False)
    
    test_metrics = DataFrame({'AUC':test_auc_epochs, 'ACC':test_acc_epochs, 'SEN':test_sen_epochs, 'SPE':test_spe_epochs})
    test_metrics.to_excel('./cross_val_results/fold'+str(k+1)+'/test_metrics.xlsx', sheet_name='sheet1', index=False)
    