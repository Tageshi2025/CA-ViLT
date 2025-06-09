'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-09 14:20:09
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-09 14:20:09
FilePath: /CA-ViLT/tasks/invariance_eval.py
Description: 因果不变性评估
'''
import torch
import torch.nn.functional as F

def invariance_loss(model, x, A_c, perturb_fn, lambda_inv=1.0):
    """
    计算因果不变性损失，评估模型的鲁棒性
    :param model: 训练好的模型
    :param x: 输入数据（图像 + 文本）
    :param A_c: 因果邻接矩阵
    :param perturb_fn: 输入扰动函数
    :param lambda_inv: 不变性损失的权重
    :return: 因果不变性损失
    """
    x_prime = perturb_fn(x)  # 对输入进行扰动
    h_real = model(x, A_c)
    h_prime = model(x_prime, A_c)
    
    loss = F.mse_loss(h_real, h_prime)  # 反事实一致性损失
    return lambda_inv * loss
