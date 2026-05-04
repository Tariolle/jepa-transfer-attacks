from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class ResnetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DSVAGenerator(nn.Module):
    """ResNet generator matching the released dSVA checkpoint key structure."""

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.resblock1 = ResnetBlock(256)
        self.resblock2 = ResnetBlock(256)
        self.resblock3 = ResnetBlock(256)
        self.resblock4 = ResnetBlock(256)
        self.resblock5 = ResnetBlock(256)
        self.resblock6 = ResnetBlock(256)
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        return self.blockf(x)


def load_dsva_generator(checkpoint: str | Path, device: torch.device) -> DSVAGenerator:
    model = DSVAGenerator()
    state_dict = torch.load(Path(checkpoint), map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Could not load dSVA generator cleanly. Missing={missing}, unexpected={unexpected}")
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def project_generator_output(
    generated: torch.Tensor,
    images: torch.Tensor,
    epsilon: float,
    output_mode: str,
) -> torch.Tensor:
    if output_mode == "adv":
        adv = generated
    elif output_mode == "delta":
        adv = images + generated
    elif output_mode == "scaled-delta":
        adv = images + epsilon * generated
    else:
        raise ValueError(f"Unsupported output mode: {output_mode}")

    adv = torch.max(torch.min(adv, images + epsilon), images - epsilon)
    return adv.clamp(0.0, 1.0)
