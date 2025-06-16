'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-12 15:19:30
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-12 16:40:40
FilePath: /CA-ViLT/model/align_vilt.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet18, ResNet18_Weights
from model.modules.cross_attention import CrossModalAttention

class AlignViltModel(nn.Module):
    def __init__(self, dim=512, num_heads=4):
        super().__init__()
        # 图像 encoder（去掉 resnet 的最后层）
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-2])  # output: B×512×7×7

        # 文本 encoder
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")

        self.patch_proj = nn.Linear(512, dim)
        self.token_proj = nn.Linear(768, dim)

        self.cross_attn = CrossModalAttention(dim, heads=num_heads)
        self.score_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, image, input_ids, attn_mask):
        # 图像编码
        feats = self.image_encoder(image)  # B×512×7×7
        B, C, H, W = feats.size()
        image_patches = feats.view(B, C, -1).permute(0, 2, 1)  # B×N×C
        image_patches = self.patch_proj(image_patches)

        # 文本编码
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        text_tokens = outputs.last_hidden_state  # B×L×768
        text_tokens = self.token_proj(text_tokens)

        # Cross-modal attention: token attends to patches
        attended_tokens, align_matrix = self.cross_attn(text_tokens, image_patches, attn_mask)

        # 对齐得分（取 CLS 向量）
        cls_repr = attended_tokens[:, 0]  # B×D
        match_logits = self.score_head(cls_repr).squeeze(-1)

        return {
            "match_logits": match_logits,      # B
            "align_matrix": align_matrix,      # B×L×N
            "text_emb": text_tokens[:, 0],     # B×D
            "image_emb": image_patches.mean(dim=1),  # B×D
        }

