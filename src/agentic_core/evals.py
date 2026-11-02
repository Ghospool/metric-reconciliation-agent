"""Minimal evaluation utilities: classification metrics and confusion matrix."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass
class ClassificationReport:
    n: int
    accuracy: float
    per_class: dict[str, dict[str, float]]
    confusion: dict[str, dict[str, int]]

    def to_markdown(self) -> str:
        lines = [f"**Accuracy:** {self.accuracy:.1%} (n={self.n})", "",
                 "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for c, m in sorted(self.per_class.items()):
            lines.append(f"| {c} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {int(m['support'])} |")
        return "\n".join(lines)


def classification_report(y_true: list[str], y_pred: list[str]) -> ClassificationReport:
    assert len(y_true) == len(y_pred) and y_true, "inputs must be equal-length and non-empty"
    labels = sorted(set(y_true) | set(y_pred))
    conf: dict[str, Counter] = defaultdict(Counter)
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1
    per: dict[str, dict[str, float]] = {}
    for c in labels:
        tp = conf[c][c]
        fp = sum(conf[t][c] for t in labels if t != c)
        fn = sum(v for p, v in conf[c].items() if p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    return ClassificationReport(len(y_true), acc, per, {k: dict(v) for k, v in conf.items()})
