import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.baseline_vilt import ViLTBaseline
from utils.dataset import ImageTextPairDataset
from utils.metrics import evaluate_match, evaluate_retrieval
import argparse, yaml, os, csv
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import datetime
import numpy as np

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



def main(config):
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # 模型与数据集初始化
    model = ViLTBaseline().to(device)
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

    # 日志与模型路径准备
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_log_{timestamp}.csv")
    best_f1 = 0.0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "loss", "acc", "f1", "precision", "recall", "auc",
            "t2i_r1", "t2i_r5", "t2i_r10", "i2t_r1", "i2t_r5", "i2t_r10", "gpu_mem(MB)"
        ])

    for epoch in range(config["train"]["epochs"]):
        model.train()
        total_loss = 0
        all_logits, all_labels = [], []
        all_image_embs, all_text_embs = [], []
        all_image_names, all_text_image_names = [], []

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].float().to(device)

            with autocast(device_type="cuda"):
                outputs = model(image, input_ids, attn_mask)
                losses = model.compute_loss(outputs, labels)
                loss = losses["total_loss"]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            all_logits.extend(outputs["match_logits"].detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_image_embs.extend(outputs["image_emb"].detach().cpu().numpy())
            all_text_embs.extend(outputs["text_emb"].detach().cpu().numpy())
            all_image_names.extend(batch["image_name"])
            all_text_image_names.extend(batch["image_name"])

        # 评估
        EVAL_LIMIT = 10000
        all_logits = all_logits[:EVAL_LIMIT]
        all_labels = all_labels[:EVAL_LIMIT]
        all_image_embs = all_image_embs[:EVAL_LIMIT]
        all_text_embs = all_text_embs[:EVAL_LIMIT]
        match_metrics = evaluate_match(np.array(all_logits), np.array(all_labels))
        retrieval_metrics = evaluate_retrieval(
            np.array(all_image_embs),
            np.array(all_text_embs),
            all_image_names,
            all_text_image_names,
        )

        avg_loss = total_loss / len(dataloader)
        gpu_mem = torch.cuda.max_memory_allocated(device) / (1024**2)

        print(
            f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | "
            f"Acc: {match_metrics['acc']:.4f} | "
            f"F1: {match_metrics['f1']:.4f} | "
            f"Precision: {match_metrics['precision']:.4f} | "
            f"Recall: {match_metrics['recall']:.4f} | "
            f"AUC: {match_metrics['auc']:.4f}"
        )

        # 保存日志
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, avg_loss,
                match_metrics["acc"], match_metrics["f1"], match_metrics["precision"],
                match_metrics["recall"], match_metrics["auc"],
                retrieval_metrics["t2i_r1"], retrieval_metrics["t2i_r5"], retrieval_metrics["t2i_r10"],
                retrieval_metrics["i2t_r1"], retrieval_metrics["i2t_r5"], retrieval_metrics["i2t_r10"],
                round(gpu_mem)
            ])

        # 保存最佳模型
        if match_metrics["f1"] > best_f1:
            best_f1 = match_metrics["f1"]
            best_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"🌟 [Best Model Saved] 当前 F1={best_f1:.4f} 已保存至 {best_path}")

        torch.cuda.reset_peak_memory_stats(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
