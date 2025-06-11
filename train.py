import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.baseline_vilt import ViLTBaseline
from utils.dataset import ImageTextPairDataset
from utils.metrics import evaluate
import argparse, yaml, os, csv
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import datetime
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(config):
    # 固定4号GPU
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # 初始化模型与优化器
    model = ViLTBaseline().to(device)
    dataset = ImageTextPairDataset(
        image_dir=config["data"]["image_dir"],
        caption_json=config["data"]["caption_json"],
        is_train=True
    )
    dataloader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # 创建日志目录 + 日志文件
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f"train_log_{timestamp}.csv")

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "acc", "f1", "gpu_mem(MB)"])

    # 训练主循环
    for epoch in range(config["train"]["epochs"]):
        model.train()
        all_preds, all_labels = [], []
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].float().to(device)

            with autocast():  # 自动混合精度
                logits = model(image, input_ids, attn_mask)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            all_preds.extend(logits.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # 评估 + 写入日志
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        metrics = evaluate(all_preds, all_labels)
        avg_loss = total_loss / len(dataloader)
        gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # 单位 MB

        print(f"[Epoch {epoch+1}] LR={config['train']['lr']} | BS={config['train']['batch_size']} | "
              f"Loss={avg_loss:.4f}, Acc={metrics['acc']:.4f}, F1={metrics['f1']:.4f}")

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, metrics["acc"], metrics["f1"], round(gpu_mem)])

        torch.cuda.reset_peak_memory_stats(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
