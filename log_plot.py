'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-11 17:15:07
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-12 11:35:58
FilePath: /CA-ViLT/log_plot.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import matplotlib.pyplot as plt
import pandas as pd
import os


def easy_painting():
    df = pd.read_csv("logs/train_log_20250611_173012.csv")

    plt.plot(df["epoch"], df["loss"], label="Loss")
    plt.plot(df["epoch"], df["acc"], label="Accuracy")
    plt.plot(df["epoch"], df["f1"], label="F1-score")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Training Curve")
    plt.grid()
    plt.show()

def plot_training_curve(log_csv_path, save_path="plots/train_curve.png"):
    df = pd.read_csv(log_csv_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["loss"], label="Total Loss")
    plt.plot(df["epoch"], df["f1"], label="F1 Score")
    plt.plot(df["epoch"], df["acc"], label="Accuracy")
    plt.plot(df["epoch"], df["precision"], label="Precision")
    plt.plot(df["epoch"], df["recall"], label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"✅ 已保存训练曲线图至: {save_path}")