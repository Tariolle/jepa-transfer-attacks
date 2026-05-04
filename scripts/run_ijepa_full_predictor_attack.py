from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.attacks.pgd import attack_pgd
from src.data.imagenet_subset import ImageNetStyleFolder, default_transform, load_class_map
from src.eval.metrics import TransferMeter, format_table
from src.models.classifiers import load_classifiers
from src.models.meta_ijepa import load_meta_ijepa_predictor
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3c: full I-JEPA predictor-objective attack with Meta's original full checkpoint."
    )
    parser.add_argument("--data-root", required=True, help="Root folder containing class subfolders.")
    parser.add_argument("--class-map", default=None, help="Optional JSON mapping folder names to ImageNet class ids.")
    parser.add_argument("--allow-folder-labels", action="store_true", help="Use local folder ids if no ImageNet id is known.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ijepa-repo", required=True, help="Path to a local clone of https://github.com/facebookresearch/ijepa.")
    parser.add_argument("--checkpoint", required=True, help="Path to Meta's full I-JEPA checkpoint .pth.tar.")
    parser.add_argument("--model-name", default="vit_huge")
    parser.add_argument("--pred-depth", type=int, default=12)
    parser.add_argument("--pred-emb-dim", type=int, default=384)
    parser.add_argument("--victims", nargs="+", default=["resnet50", "convnext_tiny", "vit_b_16"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--target-block-size", type=int, default=7)
    parser.add_argument("--context-scale", type=float, default=0.85, help="Fraction of non-target patches kept as context.")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--step-size", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--no-random-start", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true", help="Use randomly initialized victims for smoke tests only.")
    parser.add_argument("--torch-home", default=str(REPO_ROOT / ".torch_cache"), help="Cache directory for torch model weights.")
    parser.add_argument("--output-csv", default="results/phase3c_ijepa_full_predictor.csv")
    return parser.parse_args()


def block_indices(image_size: int, patch_size: int, block_size: int, batch_size: int, device: torch.device) -> torch.Tensor:
    grid_size = image_size // patch_size
    if image_size % patch_size != 0:
        raise ValueError("image-size must be divisible by patch-size")
    if not 1 <= block_size <= grid_size:
        raise ValueError("target-block-size must be in [1, image_size / patch_size]")

    start = (grid_size - block_size) // 2
    grid = torch.arange(grid_size * grid_size, device=device).view(grid_size, grid_size)
    indices = grid[start : start + block_size, start : start + block_size].flatten()
    return indices.unsqueeze(0).repeat(batch_size, 1)


def context_indices(
    image_size: int,
    patch_size: int,
    target_indices: torch.Tensor,
    context_scale: float,
    device: torch.device,
) -> torch.Tensor:
    grid_size = image_size // patch_size
    num_patches = grid_size * grid_size
    keep_count = max(1, int(round(num_patches * context_scale)))
    rows = []
    all_indices = torch.arange(num_patches, device=device)
    for targets in target_indices:
        keep = all_indices[~torch.isin(all_indices, targets)]
        rows.append(keep[: min(keep_count, keep.numel())])
    min_len = min(row.numel() for row in rows)
    return torch.stack([row[:min_len] for row in rows], dim=0)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", args.torch_home)
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"Using device: {device}", flush=True)
    print(f"Loading full I-JEPA checkpoint from {args.checkpoint}", flush=True)
    ijepa = load_meta_ijepa_predictor(
        ijepa_repo=args.ijepa_repo,
        checkpoint=args.checkpoint,
        device=device,
        model_name=args.model_name,
        image_size=args.image_size,
        patch_size=args.patch_size,
        pred_depth=args.pred_depth,
        pred_emb_dim=args.pred_emb_dim,
    )
    print("Loaded full I-JEPA encoder, predictor, and target encoder.", flush=True)
    print(f"Loading victim classifiers: {', '.join(args.victims)}", flush=True)
    classifiers = load_classifiers(args.victims, device=device, pretrained=not args.no_pretrained)
    print("Loaded victim classifiers.", flush=True)

    print(f"Loading dataset from {args.data_root}", flush=True)
    dataset = ImageNetStyleFolder(
        root=args.data_root,
        transform=default_transform(image_size=args.image_size),
        limit=args.limit,
        class_map=load_class_map(args.class_map),
        allow_folder_labels=args.allow_folder_labels,
    )
    print(f"Loaded {len(dataset)} images.", flush=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    meters = {name: TransferMeter() for name in args.victims}

    for images, labels, _paths in tqdm(loader, desc="Full I-JEPA predictor PGD"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        target_masks = block_indices(args.image_size, args.patch_size, args.target_block_size, images.shape[0], device)
        context_masks = context_indices(args.image_size, args.patch_size, target_masks, args.context_scale, device)

        clean_targets = ijepa.target_tokens(images, target_masks).detach()

        def loss_fn(images_adv: torch.Tensor) -> torch.Tensor:
            return ijepa.predictive_loss(images_adv, clean_targets, context_masks, target_masks)

        adv = attack_pgd(
            images,
            loss_fn,
            epsilon=args.epsilon,
            step_size=args.step_size,
            steps=args.steps,
            random_start=not args.no_random_start,
        )

        with torch.no_grad():
            for name, model in classifiers.items():
                meters[name].update(model(images), model(adv), labels)

    rows = [meters[name].as_row(name) for name in args.victims]
    print(format_table(rows))
    mean_transfer = sum(row["attack_success_rate"] for row in rows) / len(rows)
    print(f"\nMean transfer success: {100 * mean_transfer:.2f}%")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics to {output_csv}")


if __name__ == "__main__":
    main()
