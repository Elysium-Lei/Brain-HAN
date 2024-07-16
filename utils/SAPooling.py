# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:12:27 2023

@author: 雷雨
"""

import torch

from torch_geometric.nn import HANConv
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.utils import softmax, to_dense_adj, dense_to_sparse

def filter_adj_single(edge_index, edge_attr, perm, num_nodes):
    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index[0], edge_index[1]
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr

def filter_adj_multi(edge_index, edge_attr, perm_dict, num_nodes, mode):
    perm_bold = perm_dict['bold']
    perm_dti = perm_dict['dti']
    
    mask_bold = perm_bold.new_full((num_nodes, ), -1)
    i_bold = torch.arange(perm_bold.size(0), dtype=torch.long, device=perm_bold.device)
    mask_bold[perm_bold] = i_bold
    
    mask_dti = perm_dti.new_full((num_nodes, ), -1)
    i_dti = torch.arange(perm_dti.size(0), dtype=torch.long, device=perm_dti.device)
    mask_dti[perm_dti] = i_dti

    row, col = edge_index[0], edge_index[1]
    if mode == 1:
        row, col = mask_bold[row], mask_dti[col]
    else:
        row, col = mask_dti[row], mask_bold[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr

def creat_heteroedge(edge_index_dict, edge_types, batch_dict):
    adj_b = to_dense_adj(edge_index_dict[edge_types[0]], batch_dict['bold'])
    adj_d = to_dense_adj(edge_index_dict[edge_types[-1]], batch_dict['dti'])

    num_batch, num_nodes, _  = adj_b.size()
    threshold = int(num_nodes * 0.1)
    adj_bd = torch.empty_like(adj_b)
    adj_db = torch.empty_like(adj_d)

    for batch in range(num_batch):
        A_bold = adj_b[batch,:,:]
        A_dti = adj_d[batch,:,:]
        hetero_edge = torch.empty_like(A_bold)
        for node in range(num_nodes):
            roi_bold = A_bold[node, :].reshape(1,-1)
            simlilarity = torch.matmul(roi_bold, A_dti) / (torch.sqrt(torch.sum(roi_bold)) * torch.sqrt(torch.sum(A_dti, dim=1)).reshape(1,-1))
            simlilarity = torch.where(torch.isnan(simlilarity), torch.full_like(simlilarity, 0), simlilarity)
            sorts, indices = torch.sort(simlilarity, descending=True)

            tmp = torch.zeros_like(simlilarity)
            tmp[0, indices[0, :threshold]] = sorts[0, :threshold]
            hetero_edge[node] = tmp
        adj_bd[batch] = hetero_edge
        adj_db[batch] = torch.t(hetero_edge)

    edge_index_dict[edge_types[1]], _ = dense_to_sparse(adj_bd)
    edge_index_dict[edge_types[2]], _ = dense_to_sparse(adj_db)

    return edge_index_dict


class SAHGPooling(torch.nn.Module):
    def __init__(self, in_channels, metadata, ratio = 0.8, min_score = None, nonlinearity = 'tanh', heads = 1):
        super().__init__()

        if isinstance(nonlinearity, str):
            nonlinearity = getattr(torch, nonlinearity)

        self.in_channels = in_channels
        self.ratio = ratio
        self.conv = HANConv(in_channels, 1, metadata, heads)
        self.min_score = min_score
        self.nonlinearity = nonlinearity

    def forward(self, x_dict, edge_index_dict, batch_dict, node_types, edge_types):
        score = self.conv(x_dict, edge_index_dict)

        score_bold = score['bold'].view(-1)
        score_dti = score['dti'].view(-1)

        perm_bold = topk(score_bold, self.ratio, batch_dict['bold'], self.min_score)
        if self.min_score is None:
            x_dict['bold'] = x_dict['bold'][perm_bold] * self.nonlinearity(score_bold[perm_bold]).view(-1, 1)
        else:
            x_dict['bold'] = x_dict['bold'][perm_bold] * softmax(score_bold[perm_bold], batch_dict['bold']).view(-1, 1)

        batch_dict['bold'] = batch_dict['bold'][perm_bold]
        edge_index_dict[edge_types[0]], _ = filter_adj_single(edge_index_dict[edge_types[0]], None, perm_bold,
                                                              num_nodes=score_bold.size(0))

        perm_dti = topk(score_dti, self.ratio, batch_dict['dti'], self.min_score)
        if self.min_score is None:
            x_dict['dti'] = x_dict['dti'][perm_dti] * self.nonlinearity(score_dti[perm_dti]).view(-1, 1)
        else:
            x_dict['dti'] = x_dict['dti'][perm_dti] * softmax(score_dti[perm_dti], batch_dict['dti']).view(-1, 1)

        batch_dict['dti'] = batch_dict['dti'][perm_dti]
        edge_index_dict[edge_types[-1]], _ = filter_adj_single(edge_index_dict[edge_types[-1]], None, perm_dti,
                                                              num_nodes=score_dti.size(0))

        # edge_index_dict = creat_heteroedge(edge_index_dict, edge_types, batch_dict)

        perm_dict = {}
        perm_dict['bold'], perm_dict['dti'] = perm_bold, perm_dti
        scores = {}
        scores['bold'], scores['dti'] = score_bold, score_dti

        bold_dti = edge_index_dict[edge_types[1]]
        edge_index, _ = filter_adj_multi(bold_dti, None, perm_dict, num_nodes=score_bold.size(0), 
                                         mode=1)
        edge_index_dict[edge_types[1]] = edge_index

        dti_bold = edge_index_dict[edge_types[2]]
        edge_index, _ = filter_adj_multi(dti_bold, None, perm_dict, num_nodes=score_dti.size(0), 
                                         mode=2)
        edge_index_dict[edge_types[2]] = edge_index

        return x_dict, edge_index_dict, batch_dict, perm_dict, scores
