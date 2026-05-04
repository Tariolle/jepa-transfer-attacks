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

from src.data.imagenet_subset import ImageNetStyleFolder, default_transform, load_class_map
from src.eval.metrics import TransferMeter, format_table
from src.models.classifiers import load_classifiers
from src.models.dsva import load_dsva_generator, project_generator_output
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the released dSVA generator checkpoint directly.")
    parser.add_argument("--data-root", required=True, help="Root folder containing class subfolders.")
    parser.add_argument("--checkpoint", default="external/models/dSVA/model.pth")
    parser.add_argument("--class-map", default=None, help="Optional JSON mapping folder names to ImageNet class ids.")
    parser.add_argument("--allow-folder-labels", action="store_true", help="Use local folder ids if no ImageNet id is known.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of images to evaluate.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--victims", nargs="+", default=["resnet50", "convnext_tiny", "vit_b_16"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epsilon", type=float, default=16 / 255)
    parser.add_argument(
        "--output-mode",
        choices=["adv", "delta", "scaled-delta"],
        default="adv",
        help="Interpret generator output as adv image, raw delta, or epsilon-scaled tanh delta.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true", help="Use randomly initialized victims for smoke tests only.")
    parser.add_argument("--torch-home", default=str(REPO_ROOT / ".torch_cache"), help="Cache directory for torch model weights.")
    parser.add_argument("--output-csv", default="results/dsva_checkpoint.csv")
    return parser.parse_args()


def make_adversarial(
    generator: torch.nn.Module,
    images: torch.Tensor,
    epsilon: float,
    output_mode: str,
) -> torch.Tensor:
    generated = generator(images)
    return project_generator_output(generated, images, epsilon, output_mode)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", args.torch_home)
    set_seed(args.seed)
    device = torch.device(args.device)

    generator = load_dsva_generator(args.checkpoint, device=device)
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
    max_delta = 0.0

    for images, labels, _paths in tqdm(loader, desc="dSVA checkpoint transfer"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        adv = make_adversarial(generator, images, args.epsilon, args.output_mode)
        max_delta = max(max_delta, float((adv - images).abs().max().item()))

        with torch.no_grad():
            for name, model in classifiers.items():
                meters[name].update(model(images), model(adv), labels)

    rows = [meters[name].as_row(name) for name in args.victims]
    print(format_table(rows))
    mean_transfer = sum(row["attack_success_rate"] for row in rows) / len(rows)
    print(f"\nMean transfer success: {100 * mean_transfer:.2f}%")
    print(f"Max L_inf delta: {max_delta:.6f}")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics to {output_csv}")


if __name__ == "__main__":
    main()
