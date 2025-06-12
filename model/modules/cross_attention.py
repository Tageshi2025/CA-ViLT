'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-12 15:20:52
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-12 15:21:00
FilePath: /CA-ViLT/model/modules/cross_attention.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, tokens, patches, attn_mask=None):
        B, L, D = tokens.size()
        N = patches.size(1)
        H = self.heads
        d = D // H

        q = self.to_q(tokens).view(B, L, H, d).transpose(1, 2)  # B×H×L×d
        k = self.to_k(patches).view(B, N, H, d).transpose(1, 2)  # B×H×N×d
        v = self.to_v(patches).view(B, N, H, d).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # B×H×L×N
        attn_weights = torch.softmax(attn_scores, dim=-1)  # token → patch

        attended = (attn_weights @ v)  # B×H×L×d
        attended = attended.transpose(1, 2).contiguous().view(B, L, D)

        return self.out(attended), attn_weights.mean(dim=1)  # return mean align matrix: B×L×N
