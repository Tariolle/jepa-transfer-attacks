from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, stdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Phase 9/10 per-seed matched results and produce per-victim summaries."
    )
    parser.add_argument(
        "--phase9-root",
        default="results/phase9_dsva_official_jepa_validation",
        help="Directory containing seed_*/phase9_seed*_summary.csv and eval CSVs.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/phase10_analysis/phase9_per_victim_summary.csv",
        help="Path for the per-victim detailed summary.",
    )
    parser.add_argument(
        "--aggregate-csv",
        default="results/phase10_analysis/phase9_per_victim_aggregate.csv",
        help="Path for the across-seed aggregate summary.",
    )
    return parser.parse_args()


def find_summaries(root: Path) -> list[Path]:
    paths = sorted(root.rglob("phase9_seed*_summary.csv"))
    if not paths:
        # Fallback: accept any *_summary.csv inside seed_* folders
        paths = sorted(root.rglob("seed_*/*_summary.csv"))
    return paths


def resolve_path(path_str: str, fallback_dir: Path) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    # Try relative to repo root (strip Colab /content/jepa-transfer-attacks/ prefix)
    rel = path
    while rel.parts and rel.parts[0] in ('', 'content', 'jepa-transfer-attacks'):
        rel = Path(*rel.parts[1:])
    if rel.exists():
        return rel
    # Try same directory as summary
    local = fallback_dir / path.name
    if local.exists():
        return local
    return path


def read_eval_csv(path: Path) -> dict[str, dict[str, str]]:
    """Return {model: row_dict} from an eval CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["model"]: row for row in rows}


def main() -> None:
    args = parse_args()
    phase9_root = Path(args.phase9_root)
    output_csv = Path(args.output_csv)
    aggregate_csv = Path(args.aggregate_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    aggregate_csv.parent.mkdir(parents=True, exist_ok=True)

    summary_paths = find_summaries(phase9_root)
    if not summary_paths:
        print(f"No summary CSVs found under {phase9_root}", file=sys.stderr)
        sys.exit(1)

    detailed_rows = []
    seed_victim_gains: dict[tuple[int, str], float] = {}

    for summary_path in summary_paths:
        # Infer seed from path or filename
        seed = None
        for part in summary_path.parts:
            if part.startswith("seed_"):
                seed = int(part.split("_", 1)[1])
                break
        if seed is None:
            # Try filename
            stem = summary_path.stem
            if "seed" in stem:
                seed_part = stem.split("seed")[1].split("_")[0]
                seed = int(seed_part)
            else:
                seed = 0

        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        control = None
        jepa = None
        for row in rows:
            lr = float(row["lr"])
            if row["run_type"] == "control" and abs(lr - 5e-5) < 1e-12:
                control = row
            if row["run_type"] == "jepa" and abs(lr - 5e-5) < 1e-12 and float(row["jepa_weight"]) == 0.05:
                jepa = row

        if control is None or jepa is None:
            print(f"Skipping {summary_path}: missing matched control or JEPA row for lr=5e-5", file=sys.stderr)
            continue

        control_eval = read_eval_csv(resolve_path(control["eval_csv"], summary_path.parent))
        jepa_eval = read_eval_csv(resolve_path(jepa["eval_csv"], summary_path.parent))

        for victim in control_eval:
            if victim not in jepa_eval:
                continue
            control_asr = float(control_eval[victim]["attack_success_rate"])
            jepa_asr = float(jepa_eval[victim]["attack_success_rate"])
            gain = jepa_asr - control_asr
            detailed_rows.append(
                {
                    "victim": victim,
                    "seed": seed,
                    "control_attack_success": control_asr,
                    "jepa_attack_success": jepa_asr,
                    "matched_gain": gain,
                }
            )
            seed_victim_gains[(seed, victim)] = gain

    if not detailed_rows:
        print("No matched rows produced. Check input paths and CSV contents.", file=sys.stderr)
        sys.exit(1)

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["victim", "seed", "control_attack_success", "jepa_attack_success", "matched_gain"],
        )
        writer.writeheader()
        writer.writerows(detailed_rows)
    print(f"Wrote per-victim summary: {output_csv}")

    # Build aggregate
    victims = sorted({row["victim"] for row in detailed_rows})
    aggregate_rows = []
    for victim in victims:
        gains = [row["matched_gain"] for row in detailed_rows if row["victim"] == victim]
        controls = [row["control_attack_success"] for row in detailed_rows if row["victim"] == victim]
        jepas = [row["jepa_attack_success"] for row in detailed_rows if row["victim"] == victim]
        aggregate_rows.append(
            {
                "victim": victim,
                "control_mean": mean(controls),
                "jepa_mean": mean(jepas),
                "gain_mean": mean(gains),
                "gain_std": stdev(gains) if len(gains) > 1 else 0.0,
                "num_seeds": len(gains),
            }
        )

    with aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["victim", "control_mean", "jepa_mean", "gain_mean", "gain_std", "num_seeds"],
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)
    print(f"Wrote aggregate summary: {aggregate_csv}")

    # Print compact summary
    print("\nPer-victim matched gains:")
    for row in aggregate_rows:
        print(
            f"  {row['victim']}: control={row['control_mean']:.6f} jepa={row['jepa_mean']:.6f} "
            f"gain={row['gain_mean']:+.6f} +/- {row['gain_std']:.6f} (n={row['num_seeds']})"
        )


if __name__ == "__main__":
    main()
