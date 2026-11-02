"""CSV loading, privacy-aware profiling and normalisation."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from agentic_core.security import redact_pii

from .money import parse_cents, parse_day
from .schemas import Mapping, Row

MAX_ROWS = 2_000_000
LOW_CARD = 12


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = []
        for i, r in enumerate(csv.DictReader(f)):
            if i >= MAX_ROWS:
                raise ValueError(f"{path}: more than {MAX_ROWS} rows; aggregate or sample first")
            rows.append({(k or "").strip(): (v or "").strip() for k, v in r.items()})
    return rows


def profile(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Column summary for the model. Free-text values are never shown; only low-cardinality columns
    (statuses, regions) list their values, and anything PII-like is masked."""
    cols: dict[str, Any] = {}
    n = len(rows)
    for c in (rows[0].keys() if rows else []):
        vals = [r[c] for r in rows]
        non_empty = [v for v in vals if v != ""]
        distinct = len(set(non_empty))
        info: dict[str, Any] = {"non_empty": len(non_empty), "distinct": distinct,
                                "null_rate": round(1 - len(non_empty) / n, 4) if n else 0}
        nums = [parse_cents(v) for v in non_empty[:2000]]
        days = [parse_day(v) for v in non_empty[:2000]]
        if non_empty and sum(x is not None for x in nums) >= 0.95 * len(nums[:2000]) and distinct > LOW_CARD:
            ok = [x for x in nums if x is not None]
            info["looks_like"] = "amount"; info["min"] = min(ok) / 100; info["max"] = max(ok) / 100
        elif non_empty and sum(d is not None for d in days) >= 0.95 * len(days) and distinct > 3:
            ok = [d for d in days if d is not None]
            info["looks_like"] = "date"; info["min"] = ok and min(ok).isoformat(); info["max"] = ok and max(ok).isoformat()
        if 0 < distinct <= LOW_CARD:
            info["values"] = {redact_pii(k).text: v for k, v in Counter(non_empty).most_common()}
        elif distinct == len(non_empty) and non_empty:
            info["looks_like"] = info.get("looks_like", "unique id or free text")
            info["example"] = redact_pii(non_empty[0]).text[:24]
        cols[c] = info
    return {"rows": n, "columns": cols}


def normalise(rows: list[dict[str, str]], m: Mapping) -> tuple[list[Row], dict[str, int]]:
    need = [m.key, m.amount] + [c for c in (m.date, m.status, m.currency) if c]
    if rows:
        missing = [c for c in need if c not in rows[0]]
        if missing:
            raise ValueError(f"unknown column(s) {missing}; available: {sorted(rows[0].keys())}")
    out: list[Row] = []
    issues = {"blank_key": 0, "bad_amount": 0, "bad_date": 0}
    for i, r in enumerate(rows, 2):                      # line numbers as in the CSV file (header = 1)
        k = r[m.key]
        if not k:
            issues["blank_key"] += 1; continue
        c = parse_cents(r[m.amount])
        if c is None:
            issues["bad_amount"] += 1; continue
        d = parse_day(r[m.date]) if m.date else None
        if m.date and d is None:
            issues["bad_date"] += 1
        out.append(Row(k, c, d, r[m.status].strip().lower() if m.status else None,
                       r[m.currency].strip().upper() if m.currency else None, i))
    return out, issues
