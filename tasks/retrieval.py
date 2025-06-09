'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-09 14:20:31
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-09 14:20:31
FilePath: /CA-ViLT/tasks/retrieval.py
Description: 图像-文本检索任务
'''
# tasks/retrieval.py
import torch
import torch.nn.functional as F

def retrieval_loss(output, target):
    """
    计算图像-文本检索的损失，使用双向 InfoNCE 损失
    :param output: 模型输出 [B, D]，图像和文本的嵌入表示
    :param target: 目标标签，图像和文本的正样本对索引
    :return: 损失值
    """
    # 计算相似度矩阵
    sim_matrix = torch.matmul(output, output.T) / output.size(-1)  # [B, B]
    
    labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)  # 正样本索引
    loss_i2t = F.cross_entropy(sim_matrix, labels)
    loss_t2i = F.cross_entropy(sim_matrix.T, labels)

    return (loss_i2t + loss_t2i) / 2
