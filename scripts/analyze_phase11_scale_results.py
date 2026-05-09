from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate JEPA weight and continuation-budget sweeps."
    )
    parser.add_argument(
        "--scale-root",
        default="results/phase11_jepa_scale",
        help="Root directory containing seed_* folders with sweep summaries.",
    )
    parser.add_argument(
        "--output-detail-csv",
        default="results/phase11_analysis/jepa_scale_detail.csv",
        help="Path for one row per run and seed.",
    )
    parser.add_argument(
        "--output-aggregate-csv",
        default="results/phase11_analysis/jepa_scale_aggregate.csv",
        help="Path for across-seed aggregate rows.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_path(path_str: str, fallback_dir: Path) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    rel = path
    while rel.parts and rel.parts[0] in ("", "content", "jepa-transfer-attacks"):
        rel = Path(*rel.parts[1:])
    if rel.exists():
        return rel
    local = fallback_dir / path.name
    if local.exists():
        return local
    return path


def read_eval_metrics(path: Path) -> dict[str, float]:
    rows = read_csv(path)
    return {row["model"]: float(row["attack_success_rate"]) for row in rows}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def main() -> None:
    args = parse_args()
    root = Path(args.scale_root)
    detail_csv = Path(args.output_detail_csv)
    aggregate_csv = Path(args.output_aggregate_csv)
    detail_csv.parent.mkdir(parents=True, exist_ok=True)
    aggregate_csv.parent.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"Scale root does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    detail_rows: list[dict[str, str | float | int]] = []
    victims: list[str] = []
    for summary_path in sorted(root.glob("seed_*/*_summary.csv")):
        rows = read_csv(summary_path)
        for row in rows:
            eval_path = resolve_path(row["eval_csv"], summary_path.parent)
            if not eval_path.exists():
                print(f"Skipping missing eval CSV: {eval_path}", file=sys.stderr)
                continue
            eval_metrics = read_eval_metrics(eval_path)
            seed = row.get("seed") or summary_path.parent.name.split("_", 1)[1]
            out = {
                "experiment": summary_path.stem.removesuffix("_summary"),
                "objectives": row.get("objectives", ""),
                "run_type": row.get("run_type", ""),
                "jepa_weight": row.get("jepa_weight", ""),
                "lr": row.get("lr", ""),
                "epochs": row.get("epochs", ""),
                "train_limit": row.get("train_limit", ""),
                "eval_limit": row.get("eval_limit", ""),
                "seed": int(seed),
                "mean_transfer_success": row.get("mean_transfer_success", ""),
            }
            for victim in eval_metrics:
                if victim not in victims:
                    victims.append(victim)
                out[victim] = eval_metrics.get(victim, "")
            detail_rows.append(out)

    if not detail_rows:
        print("No sweep rows found.", file=sys.stderr)
        sys.exit(1)

    detail_fields = [
        "experiment",
        "objectives",
        "run_type",
        "jepa_weight",
        "lr",
        "epochs",
        "train_limit",
        "eval_limit",
        "seed",
        "mean_transfer_success",
        *victims,
    ]
    with detail_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)

    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, str | float | int]]] = {}
    for row in detail_rows:
        key = (
            str(row["objectives"]),
            str(row["run_type"]),
            str(row["jepa_weight"]),
            str(row["lr"]),
            str(row["epochs"]),
            str(row["train_limit"]),
        )
        groups.setdefault(key, []).append(row)

    aggregate_rows = []
    for key, rows in sorted(groups.items()):
        objective, run_type, jepa_weight, lr, epochs, train_limit = key
        transfers = [float(row["mean_transfer_success"]) for row in rows if row["mean_transfer_success"] != ""]
        out = {
            "objectives": objective,
            "run_type": run_type,
            "jepa_weight": jepa_weight,
            "lr": lr,
            "epochs": epochs,
            "train_limit": train_limit,
            "num_seeds": len(rows),
            "mean_transfer_success": f"{mean(transfers):.6f}" if transfers else "",
            "std_transfer_success": f"{stdev(transfers):.6f}" if transfers else "",
        }
        for victim in victims:
            values = [float(row[victim]) for row in rows if row.get(victim, "") != ""]
            out[f"{victim}_mean"] = f"{mean(values):.6f}" if values else ""
            out[f"{victim}_std"] = f"{stdev(values):.6f}" if values else ""
        aggregate_rows.append(out)

    aggregate_fields = [
        "objectives",
        "run_type",
        "jepa_weight",
        "lr",
        "epochs",
        "train_limit",
        "num_seeds",
        "mean_transfer_success",
        "std_transfer_success",
    ]
    for victim in victims:
        aggregate_fields.extend([f"{victim}_mean", f"{victim}_std"])

    with aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"Wrote detail CSV: {detail_csv}")
    print(f"Wrote aggregate CSV: {aggregate_csv}")


if __name__ == "__main__":
    main()
