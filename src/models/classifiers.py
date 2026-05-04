from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torchvision import models


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetClassifier(nn.Module):
    def __init__(self, model: nn.Module, mean: Iterable[float] = IMAGENET_MEAN, std: Iterable[float] = IMAGENET_STD):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.mean) / self.std
        return self.model(images)


@dataclass(frozen=True)
class LoadedClassifier:
    name: str
    model: ImageNetClassifier


def load_torchvision_classifier(name: str, pretrained: bool = True) -> ImageNetClassifier:
    weights = "DEFAULT" if pretrained else None
    try:
        model = models.get_model(name, weights=weights)
    except Exception as exc:
        raise ValueError(f"Could not load torchvision model '{name}'.") from exc

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return ImageNetClassifier(model)


def load_classifiers(names: list[str], device: torch.device, pretrained: bool = True) -> dict[str, ImageNetClassifier]:
    classifiers = {}
    for name in names:
        model = load_torchvision_classifier(name, pretrained=pretrained).to(device)
        model.eval()
        classifiers[name] = model
    return classifiers
