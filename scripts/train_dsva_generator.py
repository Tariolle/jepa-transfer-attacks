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
from src.models.ssl_encoders import load_hf_ijepa_encoder, load_hf_vit_facet_encoder, load_timm_vit_block_feature_encoder
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dSVA-style DINO/MAE generator.")
    parser.add_argument("--data-root", required=True, help="Training root containing class subfolders.")
    parser.add_argument("--class-map", default=None, help="Optional JSON mapping folder names to ImageNet class ids.")
    parser.add_argument("--allow-folder-labels", action="store_true", help="Use local folder ids if no ImageNet id is known.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of training images.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dino-backend", choices=["hf", "timm"], default="hf")
    parser.add_argument("--dino-model", default="facebook/dino-vitb16")
    parser.add_argument("--dino-block-index", type=int, default=10)
    parser.add_argument("--dino-facet", choices=["block", "q", "k", "v"], default="k")
    parser.add_argument("--dino-weight", type=float, default=0.5)
    parser.add_argument("--mae-model", default="facebook/vit-mae-base")
    parser.add_argument("--mae-block-index", type=int, default=10)
    parser.add_argument("--mae-facet", choices=["block", "q", "k", "v"], default="q")
    parser.add_argument("--mae-weight", type=float, default=0.5)
    parser.add_argument("--enable-jepa", action="store_true", help="Add I-JEPA encoder feature disruption to the dSVA loss.")
    parser.add_argument("--jepa-model", default="facebook/ijepa_vith14_1k")
    parser.add_argument("--jepa-weight", type=float, default=0.25)
    parser.add_argument("--disable-dino", action="store_true")
    parser.add_argument("--disable-mae", action="store_true")
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
    parser.add_argument("--output-checkpoint", default="results/dsva_retrained_generator.pth")
    parser.add_argument("--log-csv", default="results/dsva_retrained_generator_log.csv")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast and GradScaler for lower memory use.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true", help="Use randomly initialized encoders for smoke tests only.")
    parser.add_argument("--torch-home", default=str(REPO_ROOT / ".torch_cache"), help="Cache directory for torch model weights.")
    parser.add_argument("--hf-cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--local-files-only", action="store_true", help="Load Hugging Face models from local cache only.")
    return parser.parse_args()


def load_generator(args: argparse.Namespace, device: torch.device) -> DSVAGenerator:
    generator = DSVAGenerator().to(device)
    if args.init_checkpoint:
        generator.load_state_dict(torch.load(args.init_checkpoint, map_location="cpu"))
    return generator


def load_dino(args: argparse.Namespace, device: torch.device):
    if args.disable_dino:
        return None
    if args.dino_backend == "timm":
        return load_timm_vit_block_feature_encoder(
            model_name=args.dino_model,
            device=device,
            pretrained=not args.no_pretrained,
            image_size=args.image_size,
            block_index=args.dino_block_index,
            facet=args.dino_facet,
            feature_mode=args.feature_mode,
        )
    return load_hf_vit_facet_encoder(
        model_name=args.dino_model,
        device=device,
        pretrained=not args.no_pretrained,
        image_size=args.image_size,
        block_index=args.dino_block_index,
        facet=args.dino_facet,
        feature_mode=args.feature_mode,
        cache_dir=args.hf_cache_dir,
        local_files_only=args.local_files_only,
    )


def load_mae(args: argparse.Namespace, device: torch.device):
    if args.disable_mae:
        return None
    return load_hf_vit_facet_encoder(
        model_name=args.mae_model,
        device=device,
        pretrained=not args.no_pretrained,
        image_size=args.image_size,
        block_index=args.mae_block_index,
        facet=args.mae_facet,
        feature_mode=args.feature_mode,
        cache_dir=args.hf_cache_dir,
        local_files_only=args.local_files_only,
    )


def load_jepa(args: argparse.Namespace, device: torch.device):
    if not args.enable_jepa:
        return None
    return load_hf_ijepa_encoder(
        model_name=args.jepa_model,
        device=device,
        pretrained=not args.no_pretrained,
        image_size=args.image_size,
        feature_mode=args.feature_mode,
        cache_dir=args.hf_cache_dir,
        local_files_only=args.local_files_only,
    )


def disruption_loss(
    adv_features: torch.Tensor,
    clean_features: torch.Tensor,
    token_loss: bool,
    distance: str,
) -> torch.Tensor:
    if token_loss:
        return token_feature_disruption_loss(adv_features, clean_features, distance=distance)
    return feature_disruption_loss(adv_features, clean_features, distance=distance)


def main() -> None:
    args = parse_args()
    if args.disable_dino and args.disable_mae and not args.enable_jepa:
        raise ValueError("At least one of DINO, MAE, or JEPA must be enabled")

    os.environ.setdefault("TORCH_HOME", args.torch_home)
    set_seed(args.seed)
    device = torch.device(args.device)

    generator = load_generator(args, device)
    dino = load_dino(args, device)
    mae = load_mae(args, device)
    jepa = load_jepa(args, device)
    if dino is not None:
        dino.eval()
    if mae is not None:
        mae.eval()
    if jepa is not None:
        jepa.eval()

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
        progress = tqdm(loader, desc=f"dSVA generator epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for images, _labels, _paths in progress:
            images = images.to(device, non_blocking=True)

            with torch.no_grad():
                clean_dino = dino(images).detach() if dino is not None else None
                clean_mae = mae(images).detach() if mae is not None else None
                clean_jepa = jepa(images).detach() if jepa is not None else None

            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                adv = project_generator_output(generator(images), images, args.epsilon, args.output_mode)
                total_disruption = torch.zeros((), device=device)
                dino_disruption = None
                mae_disruption = None
                jepa_disruption = None

                if dino is not None and clean_dino is not None:
                    dino_disruption = disruption_loss(dino(adv), clean_dino, args.token_loss, args.distance)
                    total_disruption = total_disruption + args.dino_weight * dino_disruption
                if mae is not None and clean_mae is not None:
                    mae_disruption = disruption_loss(mae(adv), clean_mae, args.token_loss, args.distance)
                    total_disruption = total_disruption + args.mae_weight * mae_disruption
                if jepa is not None and clean_jepa is not None:
                    jepa_disruption = disruption_loss(jepa(adv), clean_jepa, args.token_loss, args.distance)
                    total_disruption = total_disruption + args.jepa_weight * jepa_disruption

                loss = -total_disruption / args.grad_accum_steps

            scaler.scale(loss).backward()
            global_step += 1
            if global_step % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            row = {
                "epoch": epoch + 1,
                "step": global_step,
                "total_disruption": float(total_disruption.detach().item()),
                "dino_disruption": float(dino_disruption.detach().item()) if dino_disruption is not None else 0.0,
                "mae_disruption": float(mae_disruption.detach().item()) if mae_disruption is not None else 0.0,
                "jepa_disruption": float(jepa_disruption.detach().item()) if jepa_disruption is not None else 0.0,
                "max_delta": float((adv.detach() - images).abs().max().item()),
            }
            rows.append(row)
            progress.set_postfix(total=f"{row['total_disruption']:.4f}", max_delta=f"{row['max_delta']:.5f}")

            if args.log_every > 0 and global_step % args.log_every == 0:
                with log_csv.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                    writer.writeheader()
                    writer.writerows(rows)

        if global_step % args.grad_accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        torch.save(generator.state_dict(), output_checkpoint)

    with log_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = vars(args).copy()
    output_checkpoint.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved generator to {output_checkpoint}")
    print(f"Saved training log to {log_csv}")


if __name__ == "__main__":
    main()
