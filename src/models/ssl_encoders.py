from __future__ import annotations

from typing import Iterable

import timm
import torch
from torch import nn

from src.models.classifiers import IMAGENET_MEAN, IMAGENET_STD


class TimmFeatureEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        image_size: int = 224,
        feature_mode: str = "cls",
        mean: Iterable[float] = IMAGENET_MEAN,
        std: Iterable[float] = IMAGENET_STD,
    ):
        super().__init__()
        self.model_name = model_name
        self.feature_mode = feature_mode
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=image_size,
        )
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.mean) / self.std

        if self.feature_mode == "model":
            return self.model(images)

        features = self.model.forward_features(images)
        if isinstance(features, dict):
            if self.feature_mode == "cls" and "x_norm_clstoken" in features:
                return features["x_norm_clstoken"]
            if self.feature_mode == "patch_mean" and "x_norm_patchtokens" in features:
                return features["x_norm_patchtokens"].mean(dim=1)
            raise ValueError(f"Unsupported feature mode '{self.feature_mode}' for dict features")

        if features.ndim == 3:
            if self.feature_mode == "cls":
                return features[:, 0]
            if self.feature_mode == "patch_mean":
                return features[:, 1:].mean(dim=1)
            if self.feature_mode == "tokens":
                return features

        if self.feature_mode in {"cls", "model"} and features.ndim == 2:
            return features

        raise ValueError(f"Unsupported feature mode '{self.feature_mode}' for feature shape {features.shape}")


def load_timm_feature_encoder(
    model_name: str,
    device: torch.device,
    pretrained: bool = True,
    image_size: int = 224,
    feature_mode: str = "cls",
) -> TimmFeatureEncoder:
    encoder = TimmFeatureEncoder(
        model_name=model_name,
        pretrained=pretrained,
        image_size=image_size,
        feature_mode=feature_mode,
    ).to(device)
    encoder.eval()
    return encoder


class HuggingFaceIJepaEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/ijepa_vith14_1k",
        pretrained: bool = True,
        image_size: int = 224,
        feature_mode: str = "patch_mean",
        mean: Iterable[float] = IMAGENET_MEAN,
        std: Iterable[float] = IMAGENET_STD,
    ):
        super().__init__()
        self.model_name = model_name
        self.feature_mode = feature_mode
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        try:
            from transformers import AutoModel, IJepaConfig, IJepaModel
        except ImportError as exc:
            raise ImportError("Install transformers to use the I-JEPA encoder backend.") from exc

        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            config = IJepaConfig(
                image_size=image_size,
                patch_size=14,
                hidden_size=192,
                num_hidden_layers=2,
                num_attention_heads=3,
                intermediate_size=768,
            )
            self.model = IJepaModel(config)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.mean) / self.std
        outputs = self.model(pixel_values=images, interpolate_pos_encoding=True)
        features = outputs.last_hidden_state

        if self.feature_mode == "cls":
            return features[:, 0]
        if self.feature_mode == "patch_mean":
            return features[:, 1:].mean(dim=1)
        if self.feature_mode == "tokens":
            return features

        raise ValueError(f"Unsupported I-JEPA feature mode: {self.feature_mode}")


def load_hf_ijepa_encoder(
    model_name: str,
    device: torch.device,
    pretrained: bool = True,
    image_size: int = 224,
    feature_mode: str = "patch_mean",
) -> HuggingFaceIJepaEncoder:
    encoder = HuggingFaceIJepaEncoder(
        model_name=model_name,
        pretrained=pretrained,
        image_size=image_size,
        feature_mode=feature_mode,
    ).to(device)
    encoder.eval()
    return encoder
