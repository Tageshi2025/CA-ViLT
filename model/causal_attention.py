'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-08 20:13:20
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-08 20:13:20
FilePath: /CA-ViLT/model/causal_attention.py
Description: causal_attention.py（Selective Causal Attention 模块）
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveCausalAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, gamma_init=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.gamma = nn.Parameter(torch.ones(num_heads) * gamma_init)

        self.gate_fn = nn.Sequential(
            nn.Linear(self.head_dim * 2, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, A_c):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # [B, H, L, D_h]

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        # compute causal bias
        causal_bias = torch.zeros_like(scores)
        for h in range(self.num_heads):
            qh, kh = Q[:, h], K[:, h]  # [B, L, D_h]
            pairwise = torch.einsum('bid,bjd->bijd', qh, kh)
            gate = self.gate_fn(pairwise).squeeze(-1)  # [B, L, L]
            causal_bias[:, h] = self.gamma[h] * A_c * gate

        scores = scores + causal_bias
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # [B, H, L, D_h]
        context = context.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(context)
