'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-09 14:16:11
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-09 14:16:11
FilePath: /CA-ViLT/model/causal_gnn.py
Description: 因果邻接矩阵生成
'''
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv  # Graph Attention Layer

class CausalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(CausalGNN, self).__init__()
        self.gat = GATConv(input_dim, hidden_dim, heads=num_heads)
        self.fc = nn.Linear(hidden_dim, 1)  # 预测因果矩阵的权重

    def forward(self, patch_feats, token_feats, edge_index):
        # 图神经网络 GAT 生成更新的因果节点表示
        features = torch.cat([patch_feats, token_feats], dim=0)
        updated_features = self.gat(features, edge_index)
        
        # 计算因果边的分数
        scores = self.fc(updated_features)
        return scores
