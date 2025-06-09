'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-09 14:20:55
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-09 14:20:56
FilePath: /CA-ViLT/utils/visualize.py
Description: 可视化
'''
import matplotlib.pyplot as plt
import numpy as np

def plot_attention_map(attn_map, img, top_k=5):
    """
    可视化图像和前 K 个关注的补丁
    :param attn_map: 模型的注意力权重 [B, L]
    :param img: 输入图像
    :param top_k: 可视化的前 K 个补丁
    """
    topk_idx = attn_map.topk(top_k, dim=-1).indices  # 获取 top K 的补丁
    fig, ax = plt.subplots(1, top_k)
    for i, idx in enumerate(topk_idx[0]):
        patch = img[idx].cpu().numpy()  # 转为 numpy 数组，显示图像
        ax[i].imshow(patch.transpose(1, 2, 0))  # 转换为 [H, W, C] 格式
        ax[i].set_title(f"Patch {idx.item()}")
        ax[i].axis('off')
    plt.show()

def plot_attention_map_with_image(image, attn_map, top_k=5):
    """
    在原图上可视化注意力区域
    :param image: 输入图像 [C, H, W]
    :param attn_map: 模型的注意力权重 [B, L]
    :param top_k: 可视化的前 K 个补丁
    """
    fig, ax = plt.subplots()
    ax.imshow(image.permute(1, 2, 0))  # 显示原图 [H, W, C]
    topk_idx = attn_map.topk(top_k, dim=-1).indices
    for i in range(top_k):
        ax.add_patch(plt.Rectangle((topk_idx[0][i].item(), 0), 50, 50, linewidth=2, edgecolor='r', facecolor='none'))
    plt.show()
