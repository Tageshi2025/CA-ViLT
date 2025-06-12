import torch
import torch.nn.functional as F

def soft_align_loss(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, labels.float())

def contrastive_loss(image_emb, text_emb, temperature=0.07):
    """
    InfoNCE-like contrastive loss for image-text matching.
    """
    image_emb = F.normalize(image_emb, dim=-1)  # B×D
    text_emb = F.normalize(text_emb, dim=-1)

    logits = image_emb @ text_emb.T  # B×B
    labels = torch.arange(logits.size(0)).to(logits.device)

    logits = logits / temperature
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2

def compute_loss(outputs, labels, alpha=0.8):
    match_loss = soft_align_loss(outputs["match_logits"], labels)
    contr_loss = contrastive_loss(outputs["image_emb"], outputs["text_emb"])
    total = alpha * match_loss + (1 - alpha) * contr_loss
    return {
        "match_loss": match_loss,
        "contrastive_loss": contr_loss,
        "total_loss": total
    }
