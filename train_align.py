import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.align_vilt import AlignViltModel
from model.losses import compute_loss
from utils.dataset import ImageTextPairDataset
from utils.metrics import evaluate_match
import argparse, yaml, os, csv
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def plot_training_curve(log_csv_path, save_path="plots/train_curve.png"):
    df = pd.read_csv(log_csv_path)
    plt.figure(figsize=(10, 6))
    for key in ["loss", "match_loss", "contrastive_loss", "f1", "acc", "precision", "recall"]:
        if key in df.columns:
            plt.plot(df["epoch"], df[key], label=key.capitalize())
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"✅ 已保存训练曲线图至: {save_path}")

def main(config):
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    model = AlignViltModel().to(device)
    dataset = ImageTextPairDataset(
        image_dir=config["data"]["image_dir"],
        caption_json=config["data"]["caption_json"],
        is_train=True,
        neg_strategy=config["train"].get("neg_strategy", "light_semantic"),
        max_samples=config["train"].get("max_samples", None)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )

    optimizer = optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    scaler = GradScaler()

    # 日志准备
    log_dir = "./logs"
    ckpt_dir = "./checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"align_train_log_{timestamp}.csv")
    plot_path = f"plots/align_train_curve_{timestamp}.png"

    best_f1 = 0.0
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "loss", "match_loss", "contrastive_loss",
            "acc", "f1", "precision", "recall", "auc", "gpu_mem(MB)"
        ])

    for epoch in range(config["train"]["epochs"]):
        model.train()
        total_loss = 0
        match_loss = 0
        contrastive_loss = 0

        all_logits, all_labels = [], []

        for batch in tqdm(dataloader, desc=f"[Epoch {epoch+1}]"):
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].float().to(device)

            with autocast(device_type="cuda"):
                outputs = model(image, input_ids, attn_mask)
                losses = compute_loss(outputs, labels)
                loss = losses["total_loss"]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += losses["total_loss"].item()
            match_loss += losses["match_loss"].item()
            contrastive_loss += losses["contrastive_loss"].item()

            all_logits.extend(outputs["match_logits"].detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # 评估指标
        avg_total_loss = total_loss / len(dataloader)
        avg_match_loss = match_loss / len(dataloader)
        avg_contrastive_loss = contrastive_loss / len(dataloader)
        match_metrics = evaluate_match(np.array(all_logits), np.array(all_labels))
        gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        print(
            f"[Epoch {epoch+1}] Loss: {avg_total_loss:.4f} | "
            f"Match Loss: {avg_match_loss:.4f} | Contrastive: {avg_contrastive_loss:.4f} | "
            f"Acc: {match_metrics['acc']:.4f} | F1: {match_metrics['f1']:.4f} | "
            f"Prec: {match_metrics['precision']:.4f} | Recall: {match_metrics['recall']:.4f} | "
            f"AUC: {match_metrics['auc']:.4f}"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, avg_total_loss, avg_match_loss, avg_contrastive_loss,
                match_metrics["acc"], match_metrics["f1"],
                match_metrics["precision"], match_metrics["recall"],
                match_metrics["auc"], round(gpu_mem)
            ])

        if match_metrics["f1"] > best_f1:
            best_f1 = match_metrics["f1"]
            best_path = os.path.join(ckpt_dir, "best_align_contrast.pt")
            torch.save(model.state_dict(), best_path)
            print(f"🌟 [Best] 当前F1={best_f1:.4f}，模型已保存至 {best_path}")

        torch.cuda.reset_peak_memory_stats(device)

    plot_training_curve(log_path, plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
