'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-09 14:15:40
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-09 14:15:41
FilePath: /CA-ViLT/model/vilt_base.py
Description: 主干模型
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class ViltCausalModel(nn.Module):
    def __init__(self, config):
        super(ViltCausalModel, self).__init__()
        
        # ViLT Base Configuration
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.img_embedding = nn.Linear(768, config['model']['hidden_dim'])  # assume image patch input
        self.num_heads = config['model']['num_heads']
        self.num_layers = config['model']['num_layers']
        self.hidden_dim = config['model']['hidden_dim']
        
        # Selective Causal Attention Layer
        self.causal_attention = SelectiveCausalAttention(config['model']['hidden_dim'], self.num_heads)
        
        # Feed-Forward layer for the final output
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

    def forward(self, images, texts, A_c):
        # 图像补丁和文本的嵌入
        img_feats = self.img_embedding(images)
        text_feats = self.bert(input_ids=texts).last_hidden_state  # 获取BERT的最后一层输出
        
        # 拼接图像和文本
        input_feats = torch.cat([img_feats, text_feats], dim=1)
        
        # 进行多层 Transformer 编码
        causal_output = self.causal_attention(input_feats, A_c)
        
        # 经过 FFN 层
        output = self.ffn(causal_output)
        return output
