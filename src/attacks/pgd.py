from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


def input_diversity(
    images: torch.Tensor,
    probability: float,
    min_resize: int | None = None,
) -> torch.Tensor:
    if probability <= 0.0 or torch.rand((), device=images.device).item() > probability:
        return images

    _batch_size, _channels, height, width = images.shape
    if height != width:
        raise ValueError("input_diversity currently expects square images")

    min_resize = min_resize or int(round(0.9 * height))
    if not 1 <= min_resize <= height:
        raise ValueError("input-diversity-min-resize must be in [1, image_size]")

    resize_to = int(torch.randint(min_resize, height + 1, (), device=images.device).item())
    resized = F.interpolate(images, size=(resize_to, resize_to), mode="bilinear", align_corners=False)

    pad_total = height - resize_to
    pad_top = int(torch.randint(0, pad_total + 1, (), device=images.device).item())
    pad_left = int(torch.randint(0, pad_total + 1, (), device=images.device).item())
    pad_bottom = pad_total - pad_top
    pad_right = pad_total - pad_left
    return F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)


def smooth_gradient(grad: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return grad
    if kernel_size % 2 == 0:
        raise ValueError("translation-kernel-size must be odd")

    channels = grad.shape[1]
    kernel = torch.ones((channels, 1, kernel_size, kernel_size), device=grad.device, dtype=grad.dtype)
    kernel = kernel / (kernel_size * kernel_size)
    padded = F.pad(grad, (kernel_size // 2,) * 4, mode="replicate")
    return F.conv2d(padded, kernel, groups=channels)


def attack_pgd(
    images: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    epsilon: float = 8 / 255,
    step_size: float = 2 / 255,
    steps: int = 10,
    random_start: bool = True,
    momentum: float = 0.0,
    input_diversity_prob: float = 0.0,
    input_diversity_min_resize: int | None = None,
    translation_kernel_size: int = 1,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    """Untargeted L_inf PGD that maximizes an arbitrary scalar loss."""
    clean = images.detach()

    if random_start:
        adv = clean + torch.empty_like(clean).uniform_(-epsilon, epsilon)
        adv = adv.clamp(clamp_min, clamp_max)
    else:
        adv = clean.clone()

    velocity = torch.zeros_like(clean)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss_images = input_diversity(
            adv,
            probability=input_diversity_prob,
            min_resize=input_diversity_min_resize,
        )
        loss = loss_fn(loss_images)
        if loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar tensor")

        grad = torch.autograd.grad(loss, adv, only_inputs=True)[0]
        grad = smooth_gradient(grad, translation_kernel_size)
        if momentum > 0.0:
            grad_norm = grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
            velocity = momentum * velocity + grad / grad_norm
            update = velocity
        else:
            update = grad

        adv = adv.detach() + step_size * update.sign()
        adv = torch.max(torch.min(adv, clean + epsilon), clean - epsilon)
        adv = adv.clamp(clamp_min, clamp_max)

    return adv.detach()
