from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGENETTE_WNID_TO_INDEX = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}


def default_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def load_class_map(path: str | None) -> dict[str, int]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    class_map = {}
    for key, value in raw.items():
        if isinstance(value, int):
            class_map[str(key)] = value
        elif isinstance(value, dict) and "index" in value:
            class_map[str(key)] = int(value["index"])
        else:
            raise ValueError("class-map values must be integers or objects with an 'index' field")
    return class_map


class ImageNetStyleFolder(Dataset):
    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        limit: int | None = None,
        class_map: dict[str, int] | None = None,
        allow_folder_labels: bool = False,
    ):
        self.root = Path(root)
        self.transform = transform or default_transform()
        self.class_map = class_map or {}
        self.allow_folder_labels = allow_folder_labels

        if not self.root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.root}")

        self.samples = self._find_samples()
        if limit is not None:
            self.samples = self.samples[:limit]
        if not self.samples:
            raise ValueError(f"No images found under {self.root}")

    def _find_samples(self) -> list[tuple[Path, int]]:
        image_paths = sorted(
            path for path in self.root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        samples = []
        fallback_labels = {}
        for path in image_paths:
            class_name = path.parent.name
            label = self._resolve_label(class_name, fallback_labels)
            samples.append((path, label))
        return samples

    def _resolve_label(self, class_name: str, fallback_labels: dict[str, int]) -> int:
        if class_name in self.class_map:
            return self.class_map[class_name]
        if class_name in IMAGENETTE_WNID_TO_INDEX:
            return IMAGENETTE_WNID_TO_INDEX[class_name]
        if class_name.isdigit():
            label = int(class_name)
            if 0 <= label <= 999:
                return label
        if self.allow_folder_labels:
            if class_name not in fallback_labels:
                fallback_labels[class_name] = len(fallback_labels)
            return fallback_labels[class_name]

        raise ValueError(
            f"Cannot map folder '{class_name}' to an ImageNet class id. "
            "Use numeric folders, ImageNette WNIDs, --class-map, or --allow-folder-labels."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label, str(path)
