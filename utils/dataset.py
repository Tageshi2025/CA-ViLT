'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-09 14:15:11
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-09 14:17:27
FilePath: /CA-ViLT/utils/dataset.py
Description: 数据加载
'''
# dataset.py
import os
import torch
import json
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

class CausalDataset(Dataset):
    def __init__(self, image_dir, annotation_file, tokenizer, patch_size=16):
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.tokenizer = tokenizer
        self.patch_size = patch_size

        # 加载 COCO 数据集的注释
        self.coco = COCO(annotation_file)
        
        # 获取所有图像的 ID
        self.img_ids = list(self.coco.imgs.keys())
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # 获取图像 ID 和注释信息
        img_id = self.img_ids[idx]
        image_info = self.coco.imgs[img_id]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # 图像预处理
        image = self.image_transform(image)
        
        # 获取图像对应的注释
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # 获取文本描述
        captions = [ann['caption'] for ann in anns]
        
        # 使用 BERT Tokenizer 处理文本
        text = captions[0]  # 选择第一个描述作为目标文本
        tokens = self.tokenizer(text, padding='max_length', max_length=40, truncation=True, return_tensors="pt")
        
        # 获取因果邻接矩阵 A_c
        a_c = self.generate_causal_matrix(captions, img_id)
        
        return {
            "image": image,
            "text": tokens['input_ids'].squeeze(),  # [L]
            "ac": a_c
        }

    def generate_causal_matrix(self, captions, img_id):
        """
        根据 COCO 注释和 ConceptNet/VG 生成因果邻接矩阵
        :param captions: 图像描述
        :param img_id: 图像 ID
        :return: 因果邻接矩阵 A_c
        """
        a_c = torch.zeros(196, 40)  # 假设图像分为 14x14 的 patches，文本最大 40 tokens
        for i, caption in enumerate(captions):
            tokens = caption.split()  # 简单拆分为单词
            for j, word in enumerate(tokens):
                # 使用 ConceptNet 或 Visual Genome 进行因果关系查找
                if word in self.conceptnet or word in self.vg:
                    a_c[i, j] = 1  # 依据共现关系设置因果矩阵
        return a_c
