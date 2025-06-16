import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os

# CSV 文件路径
csv_files = {
    "Baseline": "/project/CA-ViLT/logs/0_multiloss_lightsemantic.csv",
    "Align": "/project/CA-ViLT/logs/1_multiloss_lightsemantic_cross_attention.csv",
    "Align + Contrast": "/project/CA-ViLT/logs/2_multiloss_lightsemantic_contrastive_learning.csv",
    "Align + Contrast + FAISS": "/project/CA-ViLT/logs/3_faiss.csv"
}

# 输出图像文件夹
output_dir = "./figures_scatter"
os.makedirs(output_dir, exist_ok=True)

# 每组指标要绘图，横轴 epoch，颜色映射 loss
metrics_groups = {
    "Loss_Accuracy": ["acc"],
    "Classification": ["f1", "precision", "recall", "auc"],
    "Text2Image": ["t2i_r1", "t2i_r5", "t2i_r10"],
    "Image2Text": ["i2t_r1", "i2t_r5", "i2t_r10"],
    "Contrastive_Loss": ["match_loss", "contrastive_loss"]
}

x_axis = "epoch"
color_by = "loss"
cmap = cm.get_cmap("viridis")

# 为每组图绘制 Gradient Scatter 图
for group_name, metrics in metrics_groups.items():
    for y_axis in metrics:
        plt.figure(figsize=(10, 6))
        has_valid_data = False
        for name, path in csv_files.items():
            if not os.path.exists(path):
                print(f"[跳过] 文件不存在: {path}")
                continue
            df = pd.read_csv(path)
            if all(col in df.columns for col in [x_axis, y_axis, color_by]):
                norm = mcolors.Normalize(vmin=df[color_by].min(), vmax=df[color_by].max())
                plt.scatter(df[x_axis], df[y_axis],
                            c=df[color_by], cmap=cmap, norm=norm,
                            label=name, s=60, edgecolor='k', alpha=0.85)
                has_valid_data = True

        if has_valid_data:
            plt.title(f"{group_name}: {y_axis} vs {x_axis} (color: {color_by})", fontsize=14)
            plt.xlabel(x_axis.capitalize(), fontsize=12)
            plt.ylabel(y_axis.upper(), fontsize=12)
            plt.colorbar(label=color_by)
            plt.legend(fontsize=9)
            plt.grid(True)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"{group_name}_{y_axis}_scatter.png")
            plt.savefig(save_path)
            print(f"[保存成功] {save_path}")
        plt.close()
