'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-09 16:22:32
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-12 15:10:45
FilePath: /CA-ViLT/utils/dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os, json, random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from collections import defaultdict
import torch

class ImageTextPairDataset(Dataset):
    def __init__(self, image_dir, caption_json, is_train=True, 
                 tokenizer_name="bert-base-uncased", neg_strategy="light_semantic", max_samples=None):
        print(f"🌀 初始化数据集，采样策略: {neg_strategy}")

        with open(caption_json, 'r') as f:
            self.data = json.load(f)

        if max_samples:
            self.data = self.data[:max_samples]

        self.image_dir = image_dir
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.is_train = is_train
        self.neg_strategy = neg_strategy
        self.img_to_caption = {item["image"]: item["caption"] for item in self.data}

        if self.neg_strategy == "light_semantic" and self.is_train:
            self._build_category_index()

    def _build_category_index(self):
        print("🔧 构建类别倒排索引中...")
        coco_annot_path = os.path.join(os.path.dirname(self.image_dir), "annotations/instances_train2017.json")
        with open(coco_annot_path, 'r') as f:
            annot = json.load(f)

        self.img_to_categories = defaultdict(list)
        self.category_to_imgs = defaultdict(list)

        cats = {c["id"]: c["name"] for c in annot["categories"]}
        img_id_to_name = {img["id"]: img["file_name"] for img in annot["images"]}

        for ann in annot["annotations"]:
            img_name = img_id_to_name.get(ann["image_id"])
            if img_name in self.img_to_caption:
                cat_name = cats[ann["category_id"]]
                self.img_to_categories[img_name].append(cat_name)
                self.category_to_imgs[cat_name].append(img_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = os.path.join(self.image_dir, sample["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if self.is_train and random.random() < 0.5:
            # 构造轻语义负样本
            image_name = sample["image"]
            candidate = None

            if self.neg_strategy == "light_semantic" and image_name in self.img_to_categories:
                for cat in self.img_to_categories[image_name]:
                    candidates = self.category_to_imgs.get(cat, [])
                    candidates = [c for c in candidates if c != image_name]
                    if candidates:
                        candidate = random.choice(candidates)
                        break

            if not candidate:
                candidate = random.choice(list(self.img_to_caption.keys()))
                while candidate == image_name:
                    candidate = random.choice(list(self.img_to_caption.keys()))

            text = self.img_to_caption[candidate]
            label = 0
        else:
            text = sample["caption"]
            label = 1

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=40
        )

        return {
            "image": image,
            "input_ids": encoded['input_ids'].squeeze(),
            "attention_mask": encoded['attention_mask'].squeeze(),
            "label": label,
            "image_name": sample["image"]
        }
