import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torchvision.models import resnet18, ResNet18_Weights  # 导入权重枚举

class ViLTBaseline(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.text_encoder = BertModel(BertConfig())
        self.vision_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 新版写法
        self.vision_encoder.fc = nn.Linear(512, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image, input_ids, attention_mask):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        vision_out = self.vision_encoder(image)
        fused = torch.cat([vision_out, text_out], dim=1)
        logits = self.classifier(fused)
        return logits.squeeze()