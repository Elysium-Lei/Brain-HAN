# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:56:06 2023

@author: 雷雨
"""

import torch
import numpy as np


from torch import nn
from torch_geometric.nn import HANConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax, to_dense_adj, to_dense_batch, dense_to_sparse

def getP(x_dict, batch_dict, node_pool, node_padding):  
    num_node = {}
    dense_x_bold, _ = to_dense_batch(x_dict['bold'], batch_dict['bold'])
    dense_x_dti, _ = to_dense_batch(x_dict['dti'], batch_dict['dti'])
    _, num_node['bold'], _ = dense_x_bold.size()
    _, num_node['dti'], _ = dense_x_dti.size()
    
    S = torch.cat([dense_x_bold, dense_x_dti], axis=1) #batch * (node_bold + node_dti) * node_out(bold or dti)
    # S_out = S
    num_batch, node_all, node_out = S.size()
    S = S.permute(0, 2, 1)                             #batch * node_out(bold or dti) * (node_bold + node_dti)
    S = S.reshape(num_batch*node_out, node_all)        #(batch * node_out) * node_all
    
    lin = Linear(node_all, num_node[node_pool]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) 
    T = lin(S)                                         #(batch * node_out(bold or dti)) * node_bold or node_dti        
    
    new_batch = np.zeros(num_batch * node_out)
    for ind in range(num_batch): 
        new_batch[range(ind*node_out, (ind+1)*node_out)] = ind
    new_batch = torch.tensor(new_batch, dtype=batch_dict[node_pool].dtype, device=batch_dict[node_pool].device)
    
    T, _ = to_dense_batch(T, new_batch)                #batch * node_out * node_bold
    T = T.permute(0, 2, 1)                             #batch * node_bold * node_out
    T = T.reshape(num_batch*num_node[node_pool], node_out)       #(batch * node_bold) * node_out
    T = softmax(T, batch_dict[node_pool], dim=0)
    T, _ = to_dense_batch(T, batch_dict[node_pool])
    
    padding = np.zeros((num_batch, num_node[node_padding], node_out))
    padding = torch.tensor(padding, dtype=T.dtype, device=T.device)
    
    if node_pool == 'bold':       
        P = torch.cat([T, padding], axis=1)
    elif node_pool == 'dti':
        P = torch.cat([padding, T], axis=1)

    return P                                        #batch * node_all * node_out


class HGATPooling(nn.Module):
    def __init__(self, in_channels, metadata, ratio = None, out_dims = None):
        super(HGATPooling, self).__init__()

        if isinstance(out_dims, dict):
            self.out_bold = out_dims['bold']
            self.out_dti = out_dims['dti']
            
            self.conv1 = HANConv(in_channels, out_dims['bold'], metadata, heads=1)
            self.conv2 = HANConv(in_channels, out_dims['dti'], metadata, heads=1)        
        
    def forward(self, x_dict, edge_index_dict, batch_dict, node_types, edge_types):       
        dense_x_bold, _ = to_dense_batch(x_dict['bold'], batch_dict['bold'])
        dense_x_dti, _ = to_dense_batch(x_dict['dti'], batch_dict['dti'])
        x = torch.cat([dense_x_bold, dense_x_dti], axis=1)
        num_batch, _, num_features = x.size()
        
        adj_b = to_dense_adj(edge_index_dict[edge_types[0]], batch_dict['bold'])
        adj_bd = to_dense_adj(edge_index_dict[edge_types[1]], batch_dict['bold'])
        adj_db = to_dense_adj(edge_index_dict[edge_types[2]], batch_dict['dti'])
        adj_d = to_dense_adj(edge_index_dict[edge_types[3]], batch_dict['dti'])
        adj = torch.cat([torch.cat([adj_b, adj_bd], axis=2), torch.cat([adj_db, adj_d], axis=2)], axis=1)        
        
        batch_bold = np.zeros(num_batch * self.out_bold)
        for ind in range(num_batch): 
            batch_bold[range(ind*self.out_bold, (ind+1)*self.out_bold)] = ind
        batch_bold = torch.tensor(batch_bold, dtype=batch_dict['bold'].dtype, device=batch_dict['bold'].device)
        
        batch_dti = np.zeros(num_batch * self.out_dti)
        for ind in range(num_batch): 
            batch_dti[range(ind*self.out_dti, (ind+1)*self.out_dti)] = ind
        batch_dti = torch.tensor(batch_dti, dtype=batch_dict['dti'].dtype, device=batch_dict['dti'].device)
        
        x_dict_bold = self.conv1(x_dict, edge_index_dict)
        P_bold = getP(x_dict_bold, batch_dict, node_pool='bold', node_padding='dti')

        x_dict_dti = self.conv2(x_dict, edge_index_dict)
        P_dti = getP(x_dict_dti, batch_dict, node_pool='dti', node_padding='bold')
        
        x_bold = torch.matmul(P_bold.permute(0, 2, 1), x)
        x_bold = x_bold.reshape(num_batch*self.out_bold, num_features)        
        x_dti = torch.matmul(P_dti.permute(0, 2, 1), x)
        x_dti = x_dti.reshape(num_batch*self.out_dti, num_features)
        
        adj_bold = torch.matmul(P_bold.permute(0, 2, 1), torch.matmul(adj, P_bold))    
        adj_bold_to_dti = torch.matmul(P_bold.permute(0, 2, 1), torch.matmul(adj, P_dti)) 
        adj_dti_to_bold = torch.matmul(P_dti.permute(0, 2, 1), torch.matmul(adj, P_bold)) 
        adj_dti = torch.matmul(P_dti.permute(0, 2, 1), torch.matmul(adj, P_dti))
        
        edge_index_b, _ = dense_to_sparse(adj_bold)
        edge_index_bd, _ = dense_to_sparse(adj_bold_to_dti)
        edge_index_db, _ = dense_to_sparse(adj_dti_to_bold)
        edge_index_d, _ = dense_to_sparse(adj_dti)
        
        x_dict['bold'] = x_bold
        x_dict['dti'] = x_dti
        edge_index_dict[edge_types[0]] = edge_index_b
        edge_index_dict[edge_types[1]] = edge_index_bd
        edge_index_dict[edge_types[2]] = edge_index_db
        edge_index_dict[edge_types[3]] = edge_index_d
        batch_dict['bold'] = batch_bold
        batch_dict['dti'] = batch_dti
        
        return x_dict, edge_index_dict, batch_dict, P_bold, P_dti