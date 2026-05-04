from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TransferMeter:
    clean_correct: int = 0
    adv_correct: int = 0
    clean_correct_adv_wrong: int = 0
    total: int = 0

    def update(self, clean_logits: torch.Tensor, adv_logits: torch.Tensor, labels: torch.Tensor) -> None:
        clean_pred = clean_logits.argmax(dim=1)
        adv_pred = adv_logits.argmax(dim=1)
        clean_ok = clean_pred.eq(labels)
        adv_ok = adv_pred.eq(labels)

        self.clean_correct += int(clean_ok.sum().item())
        self.adv_correct += int(adv_ok.sum().item())
        self.clean_correct_adv_wrong += int((clean_ok & ~adv_ok).sum().item())
        self.total += int(labels.numel())

    @property
    def clean_accuracy(self) -> float:
        return self.clean_correct / self.total if self.total else 0.0

    @property
    def adv_accuracy(self) -> float:
        return self.adv_correct / self.total if self.total else 0.0

    @property
    def accuracy_drop(self) -> float:
        return self.clean_accuracy - self.adv_accuracy

    @property
    def attack_success_rate(self) -> float:
        return self.clean_correct_adv_wrong / self.clean_correct if self.clean_correct else 0.0

    def as_row(self, name: str) -> dict[str, float | str | int]:
        return {
            "model": name,
            "n": self.total,
            "clean_acc": self.clean_accuracy,
            "adv_acc": self.adv_accuracy,
            "acc_drop": self.accuracy_drop,
            "attack_success_rate": self.attack_success_rate,
        }


def format_table(rows: list[dict[str, float | str | int]]) -> str:
    headers = ["model", "n", "clean_acc", "adv_acc", "acc_drop", "attack_success_rate"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---:", "---:", "---:", "---:", "---:"]) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{100 * value:.2f}%")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
