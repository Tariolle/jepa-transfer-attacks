from __future__ import annotations

from typing import Callable

import torch


def attack_pgd(
    images: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    epsilon: float = 8 / 255,
    step_size: float = 2 / 255,
    steps: int = 10,
    random_start: bool = True,
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

    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = loss_fn(adv)
        if loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar tensor")

        grad = torch.autograd.grad(loss, adv, only_inputs=True)[0]
        adv = adv.detach() + step_size * grad.sign()
        adv = torch.max(torch.min(adv, clean + epsilon), clean - epsilon)
        adv = adv.clamp(clamp_min, clamp_max)

    return adv.detach()
