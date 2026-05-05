from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Phase 10 objective ablation results."
    )
    parser.add_argument(
        "--ablation-root",
        default="results/phase10_objective_ablations",
        help="Root directory containing seed_* folders with ablation summaries.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/phase10_analysis/objective_ablation_aggregate.csv",
        help="Path for the aggregate CSV.",
    )
    return parser.parse_args()


def read_summary_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_eval_csv(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["model"]: float(row["attack_success_rate"]) for row in rows}


def main() -> None:
    args = parse_args()
    root = Path(args.ablation_root)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("seed_"))
    if not seed_dirs:
        print(f"No seed directories found under {root}", file=sys.stderr)
        sys.exit(1)

    victims = ["resnet50", "convnext_tiny", "vit_b_16"]
    aggregate_rows = []

    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_", 1)[1])
        summary_files = sorted(seed_dir.glob("*_summary.csv"))
        for summary_path in summary_files:
            rows = read_summary_csv(summary_path)
            for row in rows:
                # Skip controls from ablation sweeps unless they are useful references.
                # We keep all rows; the user can filter later.
                eval_path = Path(row["eval_csv"])
                if not eval_path.exists():
                    continue
                eval_data = read_eval_csv(eval_path)
                out_row = {
                    "objectives": row.get("objectives", ""),
                    "run_type": row["run_type"],
                    "jepa_weight": row["jepa_weight"],
                    "lr": row["lr"],
                    "seed": seed,
                    "mean_transfer_success": row.get("mean_transfer_success", ""),
                }
                for victim in victims:
                    out_row[victim] = eval_data.get(victim, "")
                aggregate_rows.append(out_row)

    if not aggregate_rows:
        print("No rows aggregated.", file=sys.stderr)
        sys.exit(1)

    fieldnames = [
        "objectives",
        "run_type",
        "jepa_weight",
        "lr",
        "seed",
        "mean_transfer_success",
        *victims,
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    print(f"Wrote ablation aggregate: {output_csv}")


if __name__ == "__main__":
    main()
