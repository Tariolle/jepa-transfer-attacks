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

from src.attacks.losses import feature_disruption_loss, token_feature_disruption_loss
from src.attacks.pgd import attack_pgd
from src.data.imagenet_subset import ImageNetStyleFolder, default_transform, load_class_map
from src.eval.metrics import TransferMeter, format_table
from src.models.classifiers import load_classifiers
from src.models.ssl_encoders import load_hf_ijepa_encoder, load_timm_feature_encoder
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: SSL feature-disruption PGD transfer baseline.")
    parser.add_argument("--data-root", required=True, help="Root folder containing class subfolders.")
    parser.add_argument("--class-map", default=None, help="Optional JSON mapping folder names to ImageNet class ids.")
    parser.add_argument("--allow-folder-labels", action="store_true", help="Use local folder ids if no ImageNet id is known.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of images to evaluate.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ssl-model", default="vit_small_patch14_dinov2")
    parser.add_argument("--encoder-backend", choices=["timm", "ijepa"], default="timm")
    parser.add_argument("--feature-mode", choices=["cls", "patch_mean", "tokens", "model"], default="cls")
    parser.add_argument("--distance", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--token-loss", action="store_true", help="Compute cosine/L2 disruption per token before averaging.")
    parser.add_argument("--victims", nargs="+", default=["resnet50", "convnext_tiny", "vit_b_16"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--step-size", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--no-random-start", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true", help="Use randomly initialized models for smoke tests only.")
    parser.add_argument("--torch-home", default=str(REPO_ROOT / ".torch_cache"), help="Cache directory for torch model weights.")
    parser.add_argument("--hf-home", default=str(REPO_ROOT / ".hf_cache"), help="Cache directory for Hugging Face model weights.")
    parser.add_argument("--output-csv", default="results/phase2_ssl_feature_attack.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", args.torch_home)
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.hf_home) / "hub"))
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.encoder_backend == "ijepa":
        encoder = load_hf_ijepa_encoder(
            model_name=args.ssl_model,
            device=device,
            pretrained=not args.no_pretrained,
            image_size=args.image_size,
            feature_mode=args.feature_mode,
        )
    else:
        encoder = load_timm_feature_encoder(
            model_name=args.ssl_model,
            device=device,
            pretrained=not args.no_pretrained,
            image_size=args.image_size,
            feature_mode=args.feature_mode,
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

    for images, labels, _paths in tqdm(loader, desc="SSL PGD transfer"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            clean_features = encoder(images).detach()

        def loss_fn(images_adv: torch.Tensor) -> torch.Tensor:
            adv_features = encoder(images_adv)
            if args.token_loss:
                return token_feature_disruption_loss(adv_features, clean_features, distance=args.distance)
            return feature_disruption_loss(adv_features, clean_features, distance=args.distance)

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
