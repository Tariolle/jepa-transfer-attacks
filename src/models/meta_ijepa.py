from __future__ import annotations

import sys
import gc
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from src.models.classifiers import IMAGENET_MEAN, IMAGENET_STD


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _load_meta_vit_module(ijepa_repo: Path):
    previous_modules = {name: module for name, module in sys.modules.items() if name == "src" or name.startswith("src.")}
    previous_path = list(sys.path)
    project_root = Path(__file__).resolve().parents[2]
    for name in list(previous_modules):
        sys.modules.pop(name, None)

    sys.path = [
        path
        for path in sys.path
        if path and Path(path).resolve() != project_root and Path(path).resolve() != project_root / "scripts"
    ]
    sys.path.insert(0, str(ijepa_repo))
    try:
        import src.models.vision_transformer as vit  # type: ignore[import-not-found]
    finally:
        sys.path = previous_path

        for name in [name for name in sys.modules if name == "src" or name.startswith("src.")]:
            if name not in previous_modules:
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)

    return vit


def apply_token_masks(tokens: torch.Tensor, masks: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """Gather patch tokens by the mask-index convention used in facebookresearch/ijepa."""
    if not isinstance(masks, list):
        masks = [masks]

    outputs = []
    for mask in masks:
        if mask.ndim != 2:
            raise ValueError(f"Expected masks with shape [B, N], got {mask.shape}")
        gather_index = mask.to(tokens.device).long().unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        outputs.append(torch.gather(tokens, dim=1, index=gather_index))
    return torch.cat(outputs, dim=0)


def repeat_interleave_batch(tokens: torch.Tensor, batch_size: int, repeat: int) -> torch.Tensor:
    chunks = torch.split(tokens, batch_size, dim=0)
    return torch.cat([chunk for chunk in chunks for _ in range(repeat)], dim=0)


class MetaIJepaPredictor(nn.Module):
    """Full I-JEPA predictor objective using Meta's original archived implementation."""

    def __init__(
        self,
        ijepa_repo: str | Path,
        checkpoint: str | Path,
        device: torch.device,
        model_name: str = "vit_huge",
        image_size: int = 224,
        patch_size: int = 14,
        pred_depth: int = 12,
        pred_emb_dim: int = 384,
        mean: Iterable[float] = IMAGENET_MEAN,
        std: Iterable[float] = IMAGENET_STD,
    ):
        super().__init__()
        self.ijepa_repo = Path(ijepa_repo).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if not self.ijepa_repo.exists():
            raise FileNotFoundError(f"I-JEPA repo not found: {self.ijepa_repo}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"I-JEPA checkpoint not found: {self.checkpoint}")

        vit = _load_meta_vit_module(self.ijepa_repo)

        encoder = vit.__dict__[model_name](img_size=[image_size], patch_size=patch_size)
        predictor = vit.__dict__["vit_predictor"](
            num_patches=encoder.patch_embed.num_patches,
            embed_dim=encoder.embed_dim,
            predictor_embed_dim=pred_emb_dim,
            depth=pred_depth,
            num_heads=encoder.num_heads,
        )
        target_encoder = vit.__dict__[model_name](img_size=[image_size], patch_size=patch_size)

        print(f"Reading checkpoint file: {self.checkpoint}", flush=True)
        try:
            checkpoint_obj = torch.load(self.checkpoint, map_location="cpu", mmap=True)
        except TypeError:
            checkpoint_obj = torch.load(self.checkpoint, map_location="cpu")
        print("Checkpoint file read; loading encoder weights.", flush=True)
        encoder.load_state_dict(_strip_module_prefix(checkpoint_obj["encoder"]))
        print("Loading predictor weights.", flush=True)
        predictor.load_state_dict(_strip_module_prefix(checkpoint_obj["predictor"]))
        print("Loading target encoder weights.", flush=True)
        target_encoder.load_state_dict(_strip_module_prefix(checkpoint_obj.get("target_encoder", checkpoint_obj["encoder"])))
        del checkpoint_obj
        gc.collect()

        self.encoder = encoder.to(device).eval()
        self.predictor = predictor.to(device).eval()
        self.target_encoder = target_encoder.to(device).eval()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        for module in (self.encoder, self.predictor, self.target_encoder):
            for param in module.parameters():
                param.requires_grad_(False)

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean) / self.std

    @torch.no_grad()
    def target_tokens(self, images: torch.Tensor, target_masks: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        tokens = self.target_encoder(self.normalize(images))
        tokens = F.layer_norm(tokens, (tokens.shape[-1],))
        batch_size = tokens.shape[0]
        tokens = apply_token_masks(tokens, target_masks)
        return repeat_interleave_batch(tokens, batch_size, repeat=len(target_masks) if isinstance(target_masks, list) else 1)

    def predict_tokens(
        self,
        images: torch.Tensor,
        context_masks: torch.Tensor | list[torch.Tensor],
        target_masks: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        context_tokens = self.encoder(self.normalize(images), context_masks)
        return self.predictor(context_tokens, context_masks, target_masks)

    def predictive_loss(
        self,
        images: torch.Tensor,
        clean_target_tokens: torch.Tensor,
        context_masks: torch.Tensor | list[torch.Tensor],
        target_masks: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        predicted_tokens = self.predict_tokens(images, context_masks, target_masks)
        return F.smooth_l1_loss(predicted_tokens, clean_target_tokens)


def load_meta_ijepa_predictor(
    ijepa_repo: str | Path,
    checkpoint: str | Path,
    device: torch.device,
    model_name: str = "vit_huge",
    image_size: int = 224,
    patch_size: int = 14,
    pred_depth: int = 12,
    pred_emb_dim: int = 384,
) -> MetaIJepaPredictor:
    model = MetaIJepaPredictor(
        ijepa_repo=ijepa_repo,
        checkpoint=checkpoint,
        device=device,
        model_name=model_name,
        image_size=image_size,
        patch_size=patch_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
    ).to(device)
    model.eval()
    return model
