# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:06:59 2023

@author: 雷雨
"""

import torch

from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gap

from utils import PairNorm

class MultiHAN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, metadata, dropout_ratio):
        super(MultiHAN, self).__init__()
        self.dropout_ratio = dropout_ratio

        self.conv1 = HANConv(in_channels, hidden_channels, metadata, heads=8)      
        self.conv2 = HANConv(hidden_channels, hidden_channels, metadata, heads=8)
        self.conv3 = HANConv(hidden_channels, hidden_channels, metadata, heads=8) 

        self.lin1 = nn.Linear(hidden_channels*4, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels//2)
        self.lin3 = nn.Linear(hidden_channels//2, out_channels)
        self.norm = PairNorm('PN-SI', scale=100)
        self.act = F.relu
        
    def forward(self, data):
        x_dict, edge_index_dict, batch_dict, node_types, edge_types = data.x_dict, data.edge_index_dict, data.batch_dict, data.node_types, data.edge_types
        
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict['bold'] = self.act(x_dict['bold'])
        x_dict['dti'] = self.act(x_dict['dti'])

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict['bold'] = self.act(x_dict['bold'])
        x_dict['dti'] = self.act(x_dict['dti'])

        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict['bold'] = self.act(x_dict['bold'])
        x_dict['dti'] = self.act(x_dict['dti'])

        x_bold = torch.cat([gmp(x_dict[node_types[0]], batch_dict[node_types[0]]), gap(x_dict[node_types[0]], batch_dict[node_types[0]])], dim=1) 
        x_dti = torch.cat([gmp(x_dict[node_types[-1]], batch_dict[node_types[-1]]), gap(x_dict[node_types[-1]], batch_dict[node_types[-1]])], dim=1)
        x_feature = self.norm(torch.cat([x_bold, x_dti], dim=1))     
        
        x_out = self.act(self.lin1(x_feature))
        x_out = F.dropout(x_out, p=self.dropout_ratio, training=self.training)
        x_out = self.act(self.lin2(x_out))
        x_out = self.lin3(x_out)
        
        return x_out