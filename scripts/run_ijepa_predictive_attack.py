from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.attacks.pgd import attack_pgd
from src.data.imagenet_subset import ImageNetStyleFolder, default_transform, load_class_map
from src.eval.metrics import TransferMeter, format_table
from src.models.classifiers import load_classifiers
from src.models.ssl_encoders import load_hf_ijepa_encoder
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3b: masked-context I-JEPA predictive-inconsistency proxy. "
            "This uses the HF I-JEPA encoder with masked target tokens; the original Meta predictor "
            "requires the 10GB full checkpoint and is not loaded here."
        )
    )
    parser.add_argument("--data-root", required=True, help="Root folder containing class subfolders.")
    parser.add_argument("--class-map", default=None, help="Optional JSON mapping folder names to ImageNet class ids.")
    parser.add_argument("--allow-folder-labels", action="store_true", help="Use local folder ids if no ImageNet id is known.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of images to evaluate.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ijepa-model", default="facebook/ijepa_vith14_1k")
    parser.add_argument("--victims", nargs="+", default=["resnet50", "convnext_tiny", "vit_b_16"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--target-block-size", type=int, default=8)
    parser.add_argument("--distance", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--step-size", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--no-random-start", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true", help="Use randomly initialized I-JEPA for smoke tests only.")
    parser.add_argument("--torch-home", default=str(REPO_ROOT / ".torch_cache"), help="Cache directory for torch model weights.")
    parser.add_argument("--hf-cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--local-files-only", action="store_true", help="Load I-JEPA from local HF cache only.")
    parser.add_argument("--output-csv", default="results/phase3b_ijepa_predictive_proxy.csv")
    return parser.parse_args()


def center_block_mask(batch_size: int, image_size: int, patch_size: int, block_size: int, device: torch.device) -> torch.Tensor:
    grid_size = image_size // patch_size
    if image_size % patch_size != 0:
        raise ValueError("image-size must be divisible by patch-size")
    if not 1 <= block_size <= grid_size:
        raise ValueError("target-block-size must be in [1, image_size / patch_size]")

    start = (grid_size - block_size) // 2
    mask = torch.zeros(grid_size, grid_size, dtype=torch.bool, device=device)
    mask[start : start + block_size, start : start + block_size] = True
    return mask.flatten().unsqueeze(0).repeat(batch_size, 1)


def masked_target_distance(
    predicted_tokens: torch.Tensor,
    clean_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    distance: str,
) -> torch.Tensor:
    if predicted_tokens.shape != clean_tokens.shape:
        raise ValueError(f"Token shapes differ: {predicted_tokens.shape} vs {clean_tokens.shape}")

    selected_pred = predicted_tokens[target_mask].view(predicted_tokens.shape[0], -1, predicted_tokens.shape[-1])
    selected_clean = clean_tokens[target_mask].view(clean_tokens.shape[0], -1, clean_tokens.shape[-1])

    if distance == "cosine":
        return (1.0 - F.cosine_similarity(selected_pred, selected_clean, dim=2)).mean()
    if distance == "l2":
        selected_pred = F.normalize(selected_pred, dim=2)
        selected_clean = F.normalize(selected_clean, dim=2)
        return (selected_pred - selected_clean).square().sum(dim=2).mean()

    raise ValueError(f"Unsupported distance: {distance}")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", args.torch_home)
    set_seed(args.seed)
    device = torch.device(args.device)

    encoder = load_hf_ijepa_encoder(
        model_name=args.ijepa_model,
        device=device,
        pretrained=not args.no_pretrained,
        image_size=args.image_size,
        feature_mode="tokens",
        use_mask_token=True,
        cache_dir=args.hf_cache_dir,
        local_files_only=args.local_files_only,
    )
    classifiers = load_classifiers(args.victims, device=device, pretrained=not args.no_pretrained)

    dataset = ImageNetStyleFolder(
        root=args.data_root,
        transform=default_transform(image_size=args.image_size),
        limit=args.limit,
        class_map=load_class_map(args.class_map),
        allow_folder_labels=args.allow_folder_labels,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    meters = {name: TransferMeter() for name in args.victims}

    for images, labels, _paths in tqdm(loader, desc="I-JEPA predictive proxy PGD"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        target_mask = center_block_mask(
            batch_size=images.shape[0],
            image_size=args.image_size,
            patch_size=args.patch_size,
            block_size=args.target_block_size,
            device=device,
        )

        with torch.no_grad():
            clean_tokens = encoder.forward_tokens(images).detach()

        def loss_fn(images_adv: torch.Tensor) -> torch.Tensor:
            predicted_tokens = encoder.forward_tokens(images_adv, bool_masked_pos=target_mask)
            return masked_target_distance(predicted_tokens, clean_tokens, target_mask, args.distance)

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
                clean_logits = model(images)
                adv_logits = model(adv)
                meters[name].update(clean_logits, adv_logits, labels)

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
