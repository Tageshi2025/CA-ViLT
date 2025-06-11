'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-11 15:08:15
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-11 15:08:18
FilePath: /CA-ViLT/utils/matrics.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from sklearn.metrics import accuracy_score, f1_score

def evaluate(preds, labels):
    preds = (preds > 0.5).astype(int)
    return {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }
