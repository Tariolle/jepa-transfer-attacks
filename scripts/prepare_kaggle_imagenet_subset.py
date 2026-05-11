from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a Phase 14-ready ImageNet subset from Kaggle's localization challenge zip."
    )
    parser.add_argument("--zip", required=True, help="Path to imagenet-object-localization-challenge.zip.")
    parser.add_argument("--output-root", default="imagenet_phase14", help="Output root with train/ and val/ folders.")
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=1,
        help="Number of training images to extract per WNID class. Default gives 1000 balanced train images.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files already present in the output tree.",
    )
    return parser.parse_args()


def copy_zip_member(zf: zipfile.ZipFile, member: str | zipfile.ZipInfo, dest: Path, skip_existing: bool) -> bool:
    if skip_existing and dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, dest.open("wb") as out:
        shutil.copyfileobj(src, out, length=1024 * 1024)
    return True


def read_synsets(zf: zipfile.ZipFile) -> list[str]:
    text = zf.read("LOC_synset_mapping.txt").decode("utf-8")
    return [line.split(" ", 1)[0] for line in text.splitlines() if line.strip()]


def read_val_labels(zf: zipfile.ZipFile) -> dict[str, str]:
    labels = {}
    text = zf.read("LOC_val_solution.csv").decode("utf-8")
    for row in csv.DictReader(text.splitlines()):
        image_id = row["ImageId"]
        wnid = row["PredictionString"].split()[0]
        labels[image_id] = wnid
    return labels


def choose_train_members(zf: zipfile.ZipFile, train_per_class: int) -> list[tuple[zipfile.ZipInfo, str]]:
    chosen_by_class: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    prefix = "ILSVRC/Data/CLS-LOC/train/"
    suffixes = (".JPEG", ".jpg", ".jpeg")

    for info in zf.infolist():
        name = info.filename
        if not name.startswith(prefix) or not name.lower().endswith(tuple(s.lower() for s in suffixes)):
            continue
        rel = name.removeprefix(prefix)
        wnid = rel.split("/", 1)[0]
        bucket = chosen_by_class[wnid]
        if len(bucket) < train_per_class:
            bucket.append(info)

    missing = [wnid for wnid, members in chosen_by_class.items() if len(members) < train_per_class]
    if missing:
        print(f"Warning: {len(missing)} classes have fewer than {train_per_class} train images.")

    members: list[tuple[zipfile.ZipInfo, str]] = []
    for wnid in sorted(chosen_by_class):
        for member in sorted(chosen_by_class[wnid], key=lambda info: info.header_offset):
            members.append((member, wnid))
    members.sort(key=lambda item: item[0].header_offset)
    return members


def choose_val_members(zf: zipfile.ZipFile, val_labels: dict[str, str]) -> list[tuple[zipfile.ZipInfo, str]]:
    prefix = "ILSVRC/Data/CLS-LOC/val/"
    selected = []
    for info in zf.infolist():
        name = info.filename
        if not name.startswith(prefix) or not name.lower().endswith(".jpeg"):
            continue
        image_id = Path(name).stem
        wnid = val_labels.get(image_id)
        if wnid:
            selected.append((info, wnid))
    selected.sort(key=lambda item: item[0].header_offset)
    return selected


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip)
    output_root = Path(args.output_root)
    train_root = output_root / "train"
    val_root = output_root / "val"
    class_map_path = output_root / "imagenet_class_map.json"

    if args.train_per_class < 1:
        raise ValueError("--train-per-class must be at least 1")

    with zipfile.ZipFile(zip_path) as zf:
        synsets = read_synsets(zf)
        class_map = {wnid: index for index, wnid in enumerate(synsets)}
        output_root.mkdir(parents=True, exist_ok=True)
        class_map_path.write_text(json.dumps(class_map, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote class map: {class_map_path}", flush=True)

        train_members = choose_train_members(zf, args.train_per_class)
        print(f"Selected {len(train_members)} balanced train images.", flush=True)
        for member, wnid in tqdm(train_members, desc="Extracting train"):
            dest = train_root / wnid / Path(member.filename).name
            copy_zip_member(zf, member, dest, args.skip_existing)

        val_labels = read_val_labels(zf)
        val_members = choose_val_members(zf, val_labels)
        print(f"Selected {len(val_members)} validation images.", flush=True)
        for member, wnid in tqdm(val_members, desc="Extracting val"):
            dest = val_root / wnid / Path(member.filename).name
            copy_zip_member(zf, member, dest, args.skip_existing)

    print(f"Train root: {train_root.resolve()}")
    print(f"Val root: {val_root.resolve()}")
    print(f"Class map: {class_map_path.resolve()}")


if __name__ == "__main__":
    main()
