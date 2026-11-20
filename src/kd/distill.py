"""Soft-target, hard-label and feature-based distillation losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_target_loss(student_logits, teacher_logits, temperature):
    """Hinton KD term, scaled by T^2 to preserve gradient magnitude."""
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)


def feature_loss(student_features, teacher_features):
    """FitNets-style hint on L2-normalised feature maps (bounded in [0, 4])."""
    s = F.normalize(student_features.flatten(1), dim=1)
    t = F.normalize(teacher_features.flatten(1), dim=1)
    return (s - t).pow(2).sum(dim=1).mean()


class FeatureAdapter(nn.Module):
    """1x1 projection + BN, with adaptive pooling to align spatial sizes."""

    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(teacher_channels),
        )

    def forward(self, student_features, teacher_features):
        out = self.projection(student_features)
        if out.shape[-2:] != teacher_features.shape[-2:]:
            out = F.adaptive_avg_pool2d(out, teacher_features.shape[-2:])
        return out


class DistillCriterion:
    """Callable combining the enabled loss terms."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, logits, y, teacher_logits=None, student_features=None,
                 teacher_features=None, adapter=None):
        hard = F.cross_entropy(logits, y)
        if teacher_logits is None:
            return hard, {"hard": hard.item()}

        c = self.cfg
        alpha = float(c.get("alpha", 0.5))
        total = 0.0
        parts = {}

        if c.get("use_hard", True):
            total = total + (1.0 - alpha) * hard
            parts["hard"] = hard
        if c.get("use_soft", True):
            soft = soft_target_loss(logits, teacher_logits, float(c.get("temperature", 4.0)))
            total = total + alpha * soft
            parts["soft"] = soft
        if c.get("use_feat", False) and student_features is not None and teacher_features is not None:
            hint = feature_loss(adapter(student_features, teacher_features), teacher_features)
            total = total + float(c.get("beta", 1.0)) * hint
            parts["feat"] = hint

        return total, {k: v.item() for k, v in parts.items()}
