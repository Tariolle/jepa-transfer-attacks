from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(model(images), labels)
