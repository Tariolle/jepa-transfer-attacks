from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(model(images), labels)


def feature_disruption_loss(
    features_adv: torch.Tensor,
    features_clean: torch.Tensor,
    distance: str = "cosine",
) -> torch.Tensor:
    if features_adv.shape != features_clean.shape:
        raise ValueError(f"Feature shapes differ: {features_adv.shape} vs {features_clean.shape}")

    features_adv = features_adv.flatten(start_dim=1)
    features_clean = features_clean.flatten(start_dim=1)

    if distance == "cosine":
        return (1.0 - F.cosine_similarity(features_adv, features_clean, dim=1)).mean()
    if distance == "l2":
        features_adv = F.normalize(features_adv, dim=1)
        features_clean = F.normalize(features_clean, dim=1)
        return (features_adv - features_clean).square().sum(dim=1).mean()

    raise ValueError(f"Unsupported feature distance: {distance}")


def token_feature_disruption_loss(
    features_adv: torch.Tensor,
    features_clean: torch.Tensor,
    distance: str = "cosine",
) -> torch.Tensor:
    if features_adv.shape != features_clean.shape:
        raise ValueError(f"Feature shapes differ: {features_adv.shape} vs {features_clean.shape}")
    if features_adv.ndim != 3:
        return feature_disruption_loss(features_adv, features_clean, distance=distance)

    if distance == "cosine":
        return (1.0 - F.cosine_similarity(features_adv, features_clean, dim=2)).mean()
    if distance == "l2":
        features_adv = F.normalize(features_adv, dim=2)
        features_clean = F.normalize(features_clean, dim=2)
        return (features_adv - features_clean).square().sum(dim=2).mean()

    raise ValueError(f"Unsupported feature distance: {distance}")
