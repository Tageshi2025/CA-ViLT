import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# =========================
# 1. CSV 路径配置
# =========================
csv_files = {
    "Baseline": "/project/CA-ViLT/logs/0_multiloss_lightsemantic.csv",
    "Align": "/project/CA-ViLT/logs/1_multiloss_lightsemantic_cross_attention.csv",
    "Align + Contrast": "/project/CA-ViLT/logs/2_multiloss_lightsemantic_contrastive_learning.csv",
    "Align + Contrast + FAISS": "/project/CA-ViLT/logs/3_faiss.csv"
}

# =========================
# 2. 候选评估指标
# =========================
candidate_metrics = ["loss", "acc", "f1", "precision", "recall", "auc",
                     "t2i_r1", "t2i_r5", "t2i_r10", "i2t_r1", "i2t_r5", "i2t_r10",
                     "match_loss", "contrastive_loss"]

# =========================
# 3. 自动选择所有文件都拥有的指标
# =========================
valid_metrics = candidate_metrics.copy()
for path in csv_files.values():
    df = pd.read_csv(path)
    valid_metrics = [m for m in valid_metrics if m in df.columns]

if not valid_metrics:
    raise ValueError("没有找到所有文件共享的指标，请检查日志文件格式是否一致")

# =========================
# 4. 提取每个实验最后一轮的结果
# =========================
results = {}
for name, path in csv_files.items():
    df = pd.read_csv(path)
    final_row = df[valid_metrics].dropna().iloc[-1]
    results[name] = final_row

results_df = pd.DataFrame(results).T

# =========================
# 5. 绘图：热力图
# =========================
plt.figure(figsize=(12, 6))

# 颜色映射方案可选："YlGnBu", "YlOrRd", "coolwarm"
sns.set(font_scale=1.1)
sns.heatmap(
    results_df,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    linewidths=0.6,
    linecolor='white',
    cbar_kws={"label": "Metric Value"},
    square=False
)

plt.title("Final Evaluation Metrics Heatmap", fontsize=16, fontweight='bold')
plt.xticks(rotation=40, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

# =========================
# 6. 保存图像
# =========================
output_path = "./figures_scatter/heatmap_final_metrics.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.show()
print(f"[✔] 成功保存热力图: {output_path}")
