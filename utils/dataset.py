'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-09 16:22:32
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-12 00:22:38
FilePath: /project/CA-ViLT/utils/dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os, json, random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

class ImageTextPairDataset(Dataset):
    def __init__(self, image_dir, caption_json, is_train=True, tokenizer_name="bert-base-uncased"):
        with open(caption_json, 'r') as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = os.path.join(self.image_dir, sample["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if self.is_train and random.random() < 0.5:
            # 构造负样本
            other = random.choice(self.data)
            while other["image"] == sample["image"]:
                other = random.choice(self.data)
            text = other["caption"]
            label = 0
        else:
            text = sample["caption"]
            label = 1

        encoded = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=40)
        return {
            "image": image,
            "input_ids": encoded['input_ids'].squeeze(),
            "attention_mask": encoded['attention_mask'].squeeze(),
            "label": label
        }
