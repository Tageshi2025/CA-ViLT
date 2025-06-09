'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-09 14:17:01
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-09 14:17:01
FilePath: /CA-ViLT/utils/eval_metrics.py
Description: 评估指标
'''
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def attention_agreement_score(attn, A_c, k=5):
    """
    计算 Attention Agreement Score（注意力与因果矩阵一致性评分）
    :param attn: [B, H, L, L] 注意力权重
    :param A_c: [B, L, L] 因果邻接矩阵
    :param k: Top-K 选择的 token 数
    :return: 平均注意力一致性评分
    """
    B, H, L, _ = attn.size()
    topk_idx = attn.topk(k, dim=-1).indices  # 获取前 k 个最大注意力的 patch-token 对
    matched = 0
    for b in range(B):
        for h in range(H):
            for i in range(L):
                for j in topk_idx[b, h, i]:
                    if A_c[b, i, j] > 0.5:  # 假设因果关系的匹配度大于 0.5 为有效匹配
                        matched += 1
    return matched / (B * H * k)  # 返回平均一致性

def counterfactual_drop(model, x, A_c):
    """
    计算 Counterfactual Drop（反事实一致性损失）
    :param model: 训练好的模型
    :param x: 输入图像-文本对
    :param A_c: 因果邻接矩阵
    :return: 反事实损失
    """
    h_real = model(x)
    A_c_masked = A_c.clone()
    A_c_masked.fill_(0)  # 模拟反事实，即将 A_c 设置为 0
    h_cf = model(x, A_c_masked)  # 反事实模型输出
    return F.mse_loss(h_real, h_cf)  # 计算真实与反事实的输出差异

def visualize_attention(attn, image, top_k=5):
    """
    可视化 Attention 权重
    :param attn: [B, H, L, L] 注意力权重
    :param image: 输入图像
    :param top_k: 可视化前 K 个 attention 对
    """
    topk_idx = attn.topk(top_k, dim=-1).indices
    fig, ax = plt.subplots(1, top_k)
    for i, idx in enumerate(topk_idx[0, 0, :]):
        patch = image[idx]
        ax[i].imshow(patch.permute(1, 2, 0))  # 转换为 RGB 图像
        ax[i].set_title(f"Token {idx.item()}")
        ax[i].axis('off')
    plt.show()
