'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-08 20:13:16
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-08 20:13:16
FilePath: /CA-ViLT/train.py
Description: train.py
'''
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model.vilt_base import ViltCausalModel
from tasks.retrieval import retrieval_loss
from utils.config import load_config
from utils.dataset import load_dataset


def train_one_epoch(model, dataloader, optimizer, config):
    model.train()
    total_loss = 0
    for batch in dataloader:
        images, texts, a_c = batch["image"], batch["text"], batch["ac"]
        output = model(images, texts, a_c)
        loss = retrieval_loss(output)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main(config):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = load_dataset(config["data"], split="train", tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)

    model = ViltCausalModel(config)
    model = model.to(config["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"])

    for epoch in range(config["train"]["epochs"]):
        loss = train_one_epoch(model, train_loader, optimizer, config)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

        if (epoch + 1) % config["train"]["save_interval"] == 0:
            torch.save(model.state_dict(), f"{config['train']['save_path']}/ca_vilt_epoch{epoch}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
