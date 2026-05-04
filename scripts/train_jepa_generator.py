from __future__ import annotations

import argparse
import csv
import json
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
from src.data.imagenet_subset import ImageNetStyleFolder, default_transform, load_class_map
from src.models.dsva import DSVAGenerator, project_generator_output
from src.models.ssl_encoders import load_hf_ijepa_encoder
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dSVA-style ResNet generator with an I-JEPA encoder loss.")
    parser.add_argument("--data-root", required=True, help="Training root containing class subfolders.")
    parser.add_argument("--class-map", default=None, help="Optional JSON mapping folder names to ImageNet class ids.")
    parser.add_argument("--allow-folder-labels", action="store_true", help="Use local folder ids if no ImageNet id is known.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of training images.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ijepa-model", default="facebook/ijepa_vith14_1k")
    parser.add_argument("--feature-mode", choices=["cls", "patch_mean", "tokens"], default="tokens")
    parser.add_argument("--distance", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--token-loss", action="store_true", help="Compute disruption per token before averaging.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epsilon", type=float, default=16 / 255)
    parser.add_argument(
        "--output-mode",
        choices=["adv", "delta", "scaled-delta"],
        default="scaled-delta",
        help="Interpret generator output during training.",
    )
    parser.add_argument("--init-checkpoint", default=None, help="Optional generator checkpoint to initialize from.")
    parser.add_argument("--output-checkpoint", default="results/jepa_generator.pth")
    parser.add_argument("--log-csv", default="results/jepa_generator_train_log.csv")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast and GradScaler for lower memory use.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true", help="Use randomly initialized I-JEPA for smoke tests only.")
    parser.add_argument("--torch-home", default=str(REPO_ROOT / ".torch_cache"), help="Cache directory for torch model weights.")
    parser.add_argument("--hf-cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--local-files-only", action="store_true", help="Load I-JEPA from local HF cache only.")
    return parser.parse_args()


def load_generator(args: argparse.Namespace, device: torch.device) -> DSVAGenerator:
    generator = DSVAGenerator().to(device)
    if args.init_checkpoint:
        state_dict = torch.load(args.init_checkpoint, map_location="cpu")
        generator.load_state_dict(state_dict)
    return generator


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", args.torch_home)
    set_seed(args.seed)
    device = torch.device(args.device)

    generator = load_generator(args, device)
    encoder = load_hf_ijepa_encoder(
        model_name=args.ijepa_model,
        device=device,
        pretrained=not args.no_pretrained,
        image_size=args.image_size,
        feature_mode=args.feature_mode,
        cache_dir=args.hf_cache_dir,
        local_files_only=args.local_files_only,
    )
    encoder.eval()

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
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    output_checkpoint = Path(args.output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.log_csv)
    log_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    global_step = 0
    for epoch in range(args.epochs):
        generator.train()
        progress = tqdm(loader, desc=f"JEPA generator epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for images, _labels, _paths in progress:
            images = images.to(device, non_blocking=True)

            with torch.no_grad():
                clean_features = encoder(images).detach()

            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                generated = generator(images)
                adv = project_generator_output(generated, images, args.epsilon, args.output_mode)
                adv_features = encoder(adv)
                if args.token_loss:
                    disruption = token_feature_disruption_loss(adv_features, clean_features, distance=args.distance)
                else:
                    disruption = feature_disruption_loss(adv_features, clean_features, distance=args.distance)

                loss = -disruption / args.grad_accum_steps

            scaler.scale(loss).backward()
            global_step += 1
            if global_step % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            max_delta = float((adv.detach() - images).abs().max().item())
            row = {
                "epoch": epoch + 1,
                "step": global_step,
                "disruption_loss": float(disruption.detach().item()),
                "max_delta": max_delta,
            }
            rows.append(row)
            progress.set_postfix(disruption=f"{row['disruption_loss']:.4f}", max_delta=f"{max_delta:.5f}")

            if args.log_every > 0 and global_step % args.log_every == 0:
                with log_csv.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                    writer.writeheader()
                    writer.writerows(rows)

        torch.save(generator.state_dict(), output_checkpoint)
        if global_step % args.grad_accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    with log_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "ijepa_model": args.ijepa_model,
        "feature_mode": args.feature_mode,
        "token_loss": args.token_loss,
        "distance": args.distance,
        "epsilon": args.epsilon,
        "output_mode": args.output_mode,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "amp": args.amp,
        "limit": args.limit,
    }
    output_checkpoint.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved generator to {output_checkpoint}")
    print(f"Saved training log to {log_csv}")


if __name__ == "__main__":
    main()
