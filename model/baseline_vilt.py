import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from torchvision.models import resnet18, ResNet18_Weights


class ViLTBaseline(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.text_encoder = BertModel(BertConfig())
        
        # 图像编码器 - 保持ResNet18
        self.vision_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # 修改图像特征维度为768，使其与文本特征维度匹配
        self.vision_encoder.fc = nn.Linear(512, 768)  # 改为768维
        
        # 对比学习投影头
        self.image_proj = nn.Linear(768, hidden_dim)  # 输入768维
        self.text_proj = nn.Linear(768, hidden_dim)   # 输入768维
        
        # 修正分类器输入维度：768(图像)+768(文本)=1536
        self.classifier = nn.Sequential(
            nn.Linear(1536, hidden_dim),  # 输入维度改为1536
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, image, input_ids, attention_mask):
        # 文本编码
        text_out = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        text_pooler = text_out.pooler_output
        
        # 图像编码
        vision_out = self.vision_encoder(image)
        
        # 多模态融合 - 768+768=1536
        fused = torch.cat([vision_out, text_pooler], dim=1)
        match_logits = self.classifier(fused).squeeze()
        
        # 对比学习特征
        image_emb = F.normalize(self.image_proj(vision_out), dim=-1)
        text_emb = F.normalize(self.text_proj(text_pooler), dim=-1)
        
        return {
            "match_logits": match_logits,
            "image_emb": image_emb,
            "text_emb": text_emb
        }

    def compute_loss(self, outputs, labels):
        # 图文匹配损失
        match_loss = F.binary_cross_entropy_with_logits(
            outputs["match_logits"], labels.float()
        )

        # 对比学习损失
        image_emb = outputs["image_emb"]
        text_emb = outputs["text_emb"]

        # 计算相似度矩阵
        logits = (text_emb @ image_emb.T) / self.temperature
        targets = torch.arange(logits.size(0)).to(logits.device)

        contrastive_loss = (
            F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
        ) / 2

        # 总损失
        total_loss = match_loss + 0.5 * contrastive_loss

        return {
            "total_loss": total_loss,
            "match_loss": match_loss,
            "contrastive_loss": contrastive_loss,
        }
