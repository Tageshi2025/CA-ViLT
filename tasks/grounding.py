'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-09 14:19:48
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-09 14:19:48
FilePath: /CA-ViLT/tasks/grounding.py
Description: 短语定位
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhraseGroundingLoss(nn.Module):
    def __init__(self):
        super(PhraseGroundingLoss, self).__init__()

    def forward(self, attn_map, phrase_mask):
        """
        计算短语定位损失
        :param attn_map: 模型的注意力权重 [B, L]
        :param phrase_mask: 目标短语的 ground truth mask [B, L]
        :return: 交叉熵损失
        """
        loss = F.binary_cross_entropy(attn_map, phrase_mask)
        return loss

def grounding_evaluation(attn_map, phrase_mask):
    """
    评估短语定位的表现
    :param attn_map: 模型的注意力权重 [B, L]
    :param phrase_mask: ground truth 短语 mask [B, L]
    :return: F1 score 和 IoU
    """
    true_pos = torch.sum(attn_map * phrase_mask)
    false_pos = torch.sum(attn_map * (1 - phrase_mask))
    false_neg = torch.sum((1 - attn_map) * phrase_mask)

    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    iou = true_pos / (true_pos + false_pos + false_neg + 1e-6)

    return f1, iou
