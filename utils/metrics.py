'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-11 15:08:15
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-12 11:57:12
FilePath: /CA-ViLT/utils/metrics.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def evaluate_match(preds, labels):
    """图文匹配评估指标"""
    preds = np.array(preds)
    labels = np.array(labels)
    pred_binary = (preds > 0.5).astype(int)

    return {
        "acc": accuracy_score(labels, pred_binary),
        "f1": f1_score(labels, pred_binary, zero_division=0),
        "precision": precision_score(labels, pred_binary, zero_division=0),
        "recall": recall_score(labels, pred_binary, zero_division=0),
        "auc": roc_auc_score(labels, preds)
    }

def evaluate_retrieval(
    image_embs, text_embs, image_names, text_image_names, top_k=(1, 5, 10)
):
    """跨模态检索评估指标"""
    # 归一化特征
    image_embs = image_embs / np.linalg.norm(image_embs, axis=1, keepdims=True)
    text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)

    # 计算相似度矩阵
    sim_matrix = image_embs @ text_embs.T

    # 初始化结果字典
    results = {}

    # 文本检索图像 (T2I)
    for k in top_k:
        results[f"t2i_r{k}"] = 0.0

    # 图像检索文本 (I2T)
    for k in top_k:
        results[f"i2t_r{k}"] = 0.0

    # 计算T2I检索
    for i in range(len(text_embs)):
        sims = sim_matrix[:, i]
        sorted_indices = np.argsort(-sims)
        target_name = text_image_names[i]

        for k in top_k:
            if target_name in [image_names[idx] for idx in sorted_indices[:k]]:
                results[f"t2i_r{k}"] += 1

    # 计算I2T检索
    for i in range(len(image_embs)):
        sims = sim_matrix[i, :]
        sorted_indices = np.argsort(-sims)
        target_name = image_names[i]

        for k in top_k:
            if target_name in [text_image_names[idx] for idx in sorted_indices[:k]]:
                results[f"i2t_r{k}"] += 1

    # 归一化为比例
    for k in top_k:
        results[f"t2i_r{k}"] /= len(text_embs)
        results[f"i2t_r{k}"] /= len(image_embs)

    return results
