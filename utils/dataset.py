import os, json, random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from tqdm import tqdm
import faiss


class ImageTextPairDataset(Dataset):
    def __init__(self, image_dir, caption_json, is_train=True, 
                 tokenizer_name="bert-base-uncased", 
                 neg_strategy="random", max_samples=None):
        self.image_dir = image_dir
        self.is_train = is_train
        self.neg_strategy = neg_strategy
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        with open(caption_json, 'r') as f:
            all_data = json.load(f)
        self.data = all_data[:max_samples] if max_samples else all_data
        print(f"📦 加载图文对: {len(self.data)} 条")

        if self.neg_strategy == "hard" and self.is_train:
            self._prepare_hard_negatives()

    def _prepare_hard_negatives(self):
        print("🧠 构造 Hard Negative 样本（FAISS加速）...")
        model = BertModel.from_pretrained("bert-base-uncased").eval().cuda()
        tokenizer = self.tokenizer
        text_features = []

        for item in tqdm(self.data, desc="编码文本"):
            inputs = tokenizer(item["caption"], return_tensors="pt", truncation=True, max_length=40, padding="max_length").to("cuda")
            with torch.no_grad():
                emb = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
            text_features.append(emb[0])

        text_features = np.stack(text_features).astype("float32")
        norms = np.linalg.norm(text_features, axis=1, keepdims=True)
        text_features = text_features / norms  # normalize for cosine

        import faiss
        dim = text_features.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity by inner product
        index.add(text_features)

        D, I = index.search(text_features, 10)  # 查找 top10 相似项（含自身）

        self.neg_index_map = {}
        for i in range(len(self.data)):
            for j in I[i][1:]:  # 跳过自身
                if self.data[j]["image"] != self.data[i]["image"]:
                    self.neg_index_map[i] = j
                    break
        print(f"✅ 完成 Hard Negative 构造，覆盖样本数: {len(self.neg_index_map)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = os.path.join(self.image_dir, sample["image"])

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except:
            image = torch.zeros(3, 224, 224)

        if self.is_train and random.random() < 0.5:
            # 负样本
            if self.neg_strategy == "hard" and hasattr(self, "neg_index_map"):
                neg_idx = self.neg_index_map.get(idx, random.randint(0, len(self.data)-1))
                text = self.data[neg_idx]["caption"]
                label = 0
            else:
                # fallback to random
                other = random.choice(self.data)
                while other["image"] == sample["image"]:
                    other = random.choice(self.data)
                text = other["caption"]
                label = 0
        else:
            # 正样本
            text = sample["caption"]
            label = 1

        encoded = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=40)
        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "label": label,
            "image_name": sample["image"]
        }
