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


def select_feature_mode(features: torch.Tensor, feature_mode: str) -> torch.Tensor:
    if features.ndim == 3:
        if feature_mode == "cls":
            return features[:, 0]
        if feature_mode == "patch_mean":
            return features[:, 1:].mean(dim=1)
        if feature_mode == "tokens":
            return features
    if feature_mode in {"cls", "model"} and features.ndim == 2:
        return features
    raise ValueError(f"Unsupported feature mode '{feature_mode}' for feature shape {features.shape}")


class TimmViTBlockFeatureEncoder(nn.Module):
    """Extract intermediate block or Q/K/V facet features from timm ViTs."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        image_size: int = 224,
        block_index: int = 10,
        facet: str = "k",
        feature_mode: str = "tokens",
        mean: Iterable[float] = IMAGENET_MEAN,
        std: Iterable[float] = IMAGENET_STD,
    ):
        super().__init__()
        self.model_name = model_name
        self.block_index = block_index
        self.facet = facet
        self.feature_mode = feature_mode
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=image_size,
        )
        if not hasattr(self.model, "blocks"):
            raise ValueError(f"Model '{model_name}' does not expose ViT blocks")
        if not 0 <= block_index < len(self.model.blocks):
            raise ValueError(f"block-index must be in [0, {len(self.model.blocks) - 1}]")
        if facet not in {"block", "q", "k", "v"}:
            raise ValueError("facet must be one of: block, q, k, v")

        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.mean) / self.std
        captured: dict[str, torch.Tensor] = {}
        block = self.model.blocks[self.block_index]

        def hook_block(_module, _inputs, output):
            captured["features"] = output

        def hook_qkv(module, _inputs, output):
            batch_size, num_tokens, three_channels = output.shape
            num_heads = getattr(block.attn, "num_heads", None)
            if num_heads is None:
                raise ValueError("Cannot infer number of attention heads from timm attention block")
            head_dim = three_channels // (3 * num_heads)
            qkv = output.view(batch_size, num_tokens, 3, num_heads, head_dim)
            facet_index = {"q": 0, "k": 1, "v": 2}[self.facet]
            captured["features"] = qkv[:, :, facet_index].reshape(batch_size, num_tokens, num_heads * head_dim)

        if self.facet == "block":
            handle = block.register_forward_hook(hook_block)
        else:
            handle = block.attn.qkv.register_forward_hook(hook_qkv)
        try:
            self.model.forward_features(images)
        finally:
            handle.remove()

        if "features" not in captured:
            raise RuntimeError("Failed to capture intermediate ViT features")
        return select_feature_mode(captured["features"], self.feature_mode)


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


def load_timm_vit_block_feature_encoder(
    model_name: str,
    device: torch.device,
    pretrained: bool = True,
    image_size: int = 224,
    block_index: int = 10,
    facet: str = "k",
    feature_mode: str = "tokens",
) -> TimmViTBlockFeatureEncoder:
    encoder = TimmViTBlockFeatureEncoder(
        model_name=model_name,
        pretrained=pretrained,
        image_size=image_size,
        block_index=block_index,
        facet=facet,
        feature_mode=feature_mode,
    ).to(device)
    encoder.eval()
    return encoder


class HfHiddenStateEncoder(nn.Module):
    """Extract intermediate hidden states from Hugging Face ViT-like SSL models."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        image_size: int = 224,
        block_index: int = 10,
        feature_mode: str = "tokens",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        mean: Iterable[float] = IMAGENET_MEAN,
        std: Iterable[float] = IMAGENET_STD,
    ):
        super().__init__()
        self.model_name = model_name
        self.block_index = block_index
        self.feature_mode = feature_mode
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError("Install transformers to use Hugging Face hidden-state encoders.") from exc

        if pretrained:
            config = AutoConfig.from_pretrained(
                model_name,
                output_hidden_states=True,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                config=config,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        else:
            config = AutoConfig.from_pretrained(
                model_name,
                output_hidden_states=True,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            self.model = AutoModel.from_config(config)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.mean) / self.std
        try:
            outputs = self.model(pixel_values=images, output_hidden_states=True, interpolate_pos_encoding=True)
        except TypeError:
            outputs = self.model(pixel_values=images, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        state_index = self.block_index + 1
        if not 0 <= state_index < len(hidden_states):
            raise ValueError(f"block-index must be in [0, {len(hidden_states) - 2}]")
        return select_feature_mode(hidden_states[state_index], self.feature_mode)


def load_hf_hidden_state_encoder(
    model_name: str,
    device: torch.device,
    pretrained: bool = True,
    image_size: int = 224,
    block_index: int = 10,
    feature_mode: str = "tokens",
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> HfHiddenStateEncoder:
    encoder = HfHiddenStateEncoder(
        model_name=model_name,
        pretrained=pretrained,
        image_size=image_size,
        block_index=block_index,
        feature_mode=feature_mode,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    ).to(device)
    encoder.eval()
    return encoder


class HfViTFacetEncoder(nn.Module):
    """Extract Q/K/V or block-output facets from Hugging Face ViT-like models."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        image_size: int = 224,
        block_index: int = 10,
        facet: str = "k",
        feature_mode: str = "tokens",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        mean: Iterable[float] = IMAGENET_MEAN,
        std: Iterable[float] = IMAGENET_STD,
    ):
        super().__init__()
        self.model_name = model_name
        self.block_index = block_index
        self.facet = facet
        self.feature_mode = feature_mode
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError("Install transformers to use Hugging Face ViT facet encoders.") from exc

        if facet not in {"block", "q", "k", "v"}:
            raise ValueError("facet must be one of: block, q, k, v")

        if pretrained:
            config = AutoConfig.from_pretrained(
                model_name,
                output_hidden_states=True,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                config=config,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        else:
            config = AutoConfig.from_pretrained(
                model_name,
                output_hidden_states=True,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            self.model = AutoModel.from_config(config)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _layers(self):
        for path in ("encoder.layer", "vit.encoder.layer"):
            module = self.model
            try:
                for part in path.split("."):
                    module = getattr(module, part)
                return module
            except AttributeError:
                continue
        raise ValueError(f"Could not locate transformer layers for {self.model_name}")

    @staticmethod
    def _attention_module(layer: nn.Module) -> nn.Module:
        attention = getattr(layer, "attention", None)
        if attention is None:
            raise ValueError("Could not locate attention module")
        return getattr(attention, "attention", attention)

    def _facet_module(self, layer: nn.Module) -> nn.Module:
        attention = self._attention_module(layer)
        names = {"q": ("query", "q"), "k": ("key", "k"), "v": ("value", "v")}[self.facet]
        for name in names:
            module = getattr(attention, name, None)
            if module is not None:
                return module
        raise ValueError(f"Could not locate {self.facet} projection for {self.model_name}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.mean) / self.std
        layers = self._layers()
        if not 0 <= self.block_index < len(layers):
            raise ValueError(f"block-index must be in [0, {len(layers) - 1}]")

        captured: dict[str, torch.Tensor] = {}
        layer = layers[self.block_index]

        def hook(_module, _inputs, output):
            captured["features"] = output[0] if isinstance(output, tuple) else output

        handle = layer.register_forward_hook(hook) if self.facet == "block" else self._facet_module(layer).register_forward_hook(hook)
        try:
            try:
                self.model(pixel_values=images, output_hidden_states=True, interpolate_pos_encoding=True)
            except TypeError:
                self.model(pixel_values=images, output_hidden_states=True)
        finally:
            handle.remove()

        if "features" not in captured:
            raise RuntimeError("Failed to capture Hugging Face ViT facet features")
        return select_feature_mode(captured["features"], self.feature_mode)


def load_hf_vit_facet_encoder(
    model_name: str,
    device: torch.device,
    pretrained: bool = True,
    image_size: int = 224,
    block_index: int = 10,
    facet: str = "k",
    feature_mode: str = "tokens",
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> HfViTFacetEncoder:
    encoder = HfViTFacetEncoder(
        model_name=model_name,
        pretrained=pretrained,
        image_size=image_size,
        block_index=block_index,
        facet=facet,
        feature_mode=feature_mode,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
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
        use_mask_token: bool = False,
        cache_dir: str | None = None,
        local_files_only: bool = False,
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
            self.model = AutoModel.from_pretrained(
                model_name,
                use_mask_token=use_mask_token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        else:
            config = IJepaConfig(
                image_size=image_size,
                patch_size=14,
                hidden_size=192,
                num_hidden_layers=2,
                num_attention_heads=3,
                intermediate_size=768,
            )
            self.model = IJepaModel(config, use_mask_token=use_mask_token)

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
            return features.mean(dim=1)
        if self.feature_mode == "tokens":
            return features

        raise ValueError(f"Unsupported I-JEPA feature mode: {self.feature_mode}")

    def forward_tokens(self, images: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> torch.Tensor:
        images = (images - self.mean) / self.std
        outputs = self.model(
            pixel_values=images,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=True,
        )
        return outputs.last_hidden_state


def load_hf_ijepa_encoder(
    model_name: str,
    device: torch.device,
    pretrained: bool = True,
    image_size: int = 224,
    feature_mode: str = "patch_mean",
    use_mask_token: bool = False,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> HuggingFaceIJepaEncoder:
    encoder = HuggingFaceIJepaEncoder(
        model_name=model_name,
        pretrained=pretrained,
        image_size=image_size,
        feature_mode=feature_mode,
        use_mask_token=use_mask_token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    ).to(device)
    encoder.eval()
    return encoder
