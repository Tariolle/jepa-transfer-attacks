from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CONFIGS = [
    "0.01:0.00002",
    "0.025:0.00002",
    "0.05:0.00001",
    "0.05:0.00002",
    "0.05:0.00005",
    "0.1:0.00002",
]


def parse_config(value: str) -> tuple[float, float]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected WEIGHT:LR, for example 0.025:0.00002")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("WEIGHT and LR must be numeric") from exc


def format_tag(value: float) -> str:
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def quote_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def run_command(command: list[str], dry_run: bool) -> None:
    print(f"\n$ {quote_command(command)}")
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def read_mean_transfer(csv_path: Path) -> float:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return sum(float(row["attack_success_rate"]) for row in rows) / len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep low-weight I-JEPA fine-tuning on top of a dSVA-style generator."
    )
    parser.add_argument("--train-root", default="imagenette2-320/train", help="Training root containing class subfolders.")
    parser.add_argument("--val-root", default="imagenette2-320/val", help="Validation root containing class subfolders.")
    parser.add_argument("--class-map", default=None, help="Optional JSON mapping folder names to ImageNet class ids.")
    parser.add_argument("--allow-folder-labels", action="store_true", help="Use local folder ids if no ImageNet id is known.")
    parser.add_argument("--init-checkpoint", default="results/dsva_retrained_eps16.pth")
    parser.add_argument("--output-dir", default="results/dsva_jepa_sweep")
    parser.add_argument("--run-prefix", default="dsva_jepa_ft")
    parser.add_argument(
        "--configs",
        nargs="+",
        type=parse_config,
        default=[parse_config(item) for item in DEFAULT_CONFIGS],
        metavar="WEIGHT:LR",
        help="Fine-tune configs. Default: %(default)s",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Training image limit.")
    parser.add_argument("--eval-limit", type=int, default=100, help="Evaluation image limit.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=16 / 255)
    parser.add_argument(
        "--output-mode",
        choices=["adv", "delta", "scaled-delta"],
        default="scaled-delta",
        help="Interpret generator output during training and evaluation.",
    )
    parser.add_argument("--victims", nargs="+", default=["resnet50", "convnext_tiny", "vit_b_16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP during training.")
    parser.add_argument("--eval-amp", action="store_true", help="Use CUDA autocast during evaluation.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for training modules.")
    parser.add_argument("--eval-compile", action="store_true", help="Use torch.compile for evaluation modules.")
    parser.add_argument("--compile-mode", default="reduce-overhead", help="torch.compile mode forwarded to train/eval scripts.")
    parser.add_argument(
        "--normalize-loss-weights",
        action="store_true",
        help="Keep total loss scale comparable when JEPA is added.",
    )
    parser.add_argument(
        "--no-controls",
        action="store_true",
        help="Do not run matched DINO+MAE continuation controls without JEPA.",
    )
    parser.add_argument("--disable-dino", action="store_true", help="Pass --disable-dino to training.")
    parser.add_argument("--disable-mae", action="store_true", help="Pass --disable-mae to training.")
    parser.add_argument(
        "--extra-train-args",
        nargs="+",
        default=[],
        help="Extra arguments forwarded to the training script.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing checkpoints/eval CSVs when present.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.run_prefix}_summary.csv"

    experiments: list[tuple[str, float, float, bool]] = []
    if not args.no_controls:
        for lr in sorted({lr for _jepa_weight, lr in args.configs}):
            experiments.append(("control", 0.0, lr, False))
    for jepa_weight, lr in args.configs:
        experiments.append(("jepa", jepa_weight, lr, True))

    def objective_label(enable_jepa: bool) -> str:
        dino = not args.disable_dino
        mae = not args.disable_mae
        parts = []
        if dino:
            parts.append("dino")
        if mae:
            parts.append("mae")
        if enable_jepa:
            parts.append("jepa")
        if not parts:
            return "none"
        return "_".join(parts)

    summary_rows = []
    for run_type, jepa_weight, lr, enable_jepa in experiments:
        if enable_jepa:
            tag = f"jw{format_tag(jepa_weight)}_lr{format_tag(lr)}"
        else:
            tag = f"control_lr{format_tag(lr)}"
        checkpoint = output_dir / f"{args.run_prefix}_{tag}.pth"
        train_log = output_dir / f"{args.run_prefix}_{tag}_train.csv"
        eval_csv = output_dir / f"{args.run_prefix}_{tag}_eval.csv"

        train_cmd = [
            sys.executable,
            "scripts/train_dsva_generator.py",
            "--data-root",
            args.train_root,
            "--limit",
            str(args.limit),
            "--batch-size",
            str(args.batch_size),
            "--grad-accum-steps",
            str(args.grad_accum_steps),
            "--epochs",
            str(args.epochs),
            "--token-loss",
            "--lr",
            str(lr),
            "--init-checkpoint",
            args.init_checkpoint,
            "--epsilon",
            str(args.epsilon),
            "--output-mode",
            args.output_mode,
            "--device",
            args.device,
            "--num-workers",
            str(args.num_workers),
            "--output-checkpoint",
            str(checkpoint),
            "--log-csv",
            str(train_log),
            "--seed",
            str(args.seed),
        ]
        if enable_jepa:
            train_cmd.extend(["--enable-jepa", "--jepa-weight", str(jepa_weight)])
        if args.normalize_loss_weights:
            train_cmd.append("--normalize-loss-weights")
        if args.disable_dino:
            train_cmd.append("--disable-dino")
        if args.disable_mae:
            train_cmd.append("--disable-mae")
        if not args.no_amp:
            train_cmd.append("--amp")
        if args.compile:
            train_cmd.extend(["--compile", "--compile-mode", args.compile_mode])
        if args.hf_cache_dir:
            train_cmd.extend(["--hf-cache-dir", args.hf_cache_dir])
        if args.local_files_only:
            train_cmd.append("--local-files-only")
        if args.class_map:
            train_cmd.extend(["--class-map", args.class_map])
        if args.allow_folder_labels:
            train_cmd.append("--allow-folder-labels")
        if args.extra_train_args:
            train_cmd.extend(args.extra_train_args)

        eval_cmd = [
            sys.executable,
            "scripts/run_dsva_checkpoint_attack.py",
            "--data-root",
            args.val_root,
            "--checkpoint",
            str(checkpoint),
            "--output-mode",
            args.output_mode,
            "--limit",
            str(args.eval_limit),
            "--batch-size",
            str(args.eval_batch_size),
            "--epsilon",
            str(args.epsilon),
            "--victims",
            *args.victims,
            "--device",
            args.device,
            "--num-workers",
            str(args.num_workers),
            "--output-csv",
            str(eval_csv),
            "--seed",
            str(args.seed),
        ]
        if args.eval_amp:
            eval_cmd.append("--amp")
        if args.eval_compile:
            eval_cmd.extend(["--compile", "--compile-mode", args.compile_mode])
        if args.class_map:
            eval_cmd.extend(["--class-map", args.class_map])
        if args.allow_folder_labels:
            eval_cmd.append("--allow-folder-labels")

        if args.skip_existing and checkpoint.exists():
            print(f"\nSkipping train for existing checkpoint: {checkpoint}")
        else:
            run_command(train_cmd, args.dry_run)

        if args.skip_existing and eval_csv.exists():
            print(f"Skipping eval for existing metrics: {eval_csv}")
        else:
            run_command(eval_cmd, args.dry_run)

        mean_transfer = "" if args.dry_run else f"{read_mean_transfer(eval_csv):.6f}"
        summary_rows.append(
            {
                "objectives": objective_label(enable_jepa),
                "run_type": run_type,
                "jepa_weight": jepa_weight,
                "lr": lr,
                "epochs": args.epochs,
                "train_limit": args.limit,
                "eval_limit": args.eval_limit,
                "seed": args.seed,
                "mean_transfer_success": mean_transfer,
                "checkpoint": str(checkpoint),
                "train_log": str(train_log),
                "eval_csv": str(eval_csv),
            }
        )

        if not args.dry_run:
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"Updated sweep summary: {summary_path}")

    if args.dry_run:
        print(f"\nDry run complete. Planned summary path: {summary_path}")
    else:
        print(f"\nSaved sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
