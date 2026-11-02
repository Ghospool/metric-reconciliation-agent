"""Numbers in model-written text must come from the bridge. Anything else is flagged, never silently kept."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from .money import fmt
from .schemas import Step

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUM = re.compile(r"(?<![\w.])[-+]?[$€£]?\d[\d,]*(?:\.\d+)?%?")


def allowed_numbers(steps: list[Step], a_rep: int, b_rep: int, residual: int, gap: int) -> set[Decimal]:
    vals: set[Decimal] = set()

    def add(x) -> None:
        vals.add(Decimal(str(x)).normalize())

    for c in (a_rep, b_rep, residual, gap):
        add(Decimal(abs(c)) / 100)
    for s in steps:
        add(Decimal(abs(s.delta_cents)) / 100); add(s.rows)
        if gap:
            add(round(abs(s.delta_cents) / abs(gap) * 100, 1))
        for v in s.notes.values():
            if isinstance(v, dict):
                for x in v.values():
                    if isinstance(x, (int, float)): add(x); add(round(x * 100, 1))
        for smp in s.samples:
            for x in smp.values():
                if isinstance(x, (int, float)): add(x)
    if gap:
        add(round(abs(residual) / abs(gap) * 100, 1))
    return vals


def unverified_numbers(text: str, allowed: set[Decimal]) -> list[str]:
    bad = []
    for tok in _NUM.findall(_DATE.sub(" ", text)):
        raw = tok.lstrip("+-$€£").rstrip("%").replace(",", "")
        try:
            d = Decimal(raw).normalize()
        except InvalidOperation:
            continue
        significant = len(raw.replace(".", "")) >= 3 or "." in raw or tok.endswith("%")
        if significant and d not in allowed:
            bad.append(tok)
    return bad


def explained_share(steps: list[Step], gap: int) -> float:
    return 1.0 if gap == 0 else 1 - abs(gap - sum(s.delta_cents for s in steps)) / abs(gap)


__all__ = ["allowed_numbers", "unverified_numbers", "explained_share", "fmt"]
