"""Deterministic probes. Each one transforms the working state toward agreement and records its effect.

Bridge convention: contribution = (change in A total) - (change in B total), so that
    B_reported - A_reported == sum(step contributions) + residual         (always, exactly)
The result depends on probe order (a waterfall); order is recorded and shown in the report."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date
from statistics import median
from typing import Any

from .schemas import Row, Step

SAMPLE_N = 5


@dataclass
class State:
    a_all: list[Row]
    b_all: list[Row]
    start: date
    end: date
    a_cur: list[Row] = field(default_factory=list)
    b_cur: list[Row] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    a_reported: int = 0
    b_reported: int = 0

    def __post_init__(self) -> None:
        inp = lambda r: r.day is not None and self.start <= r.day <= self.end
        self.a_cur = [copy.copy(r) for r in self.a_all if inp(r)]
        self.b_cur = [copy.copy(r) for r in self.b_all if inp(r)]
        self.a_reported, self.b_reported = tot(self.a_cur), tot(self.b_cur)

    def in_period(self, r: Row) -> bool:
        return r.day is not None and self.start <= r.day <= self.end

    @property
    def residual(self) -> int:
        return tot(self.b_cur) - tot(self.a_cur)

    @property
    def gap(self) -> int:
        return self.b_reported - self.a_reported


def tot(rows: list[Row]) -> int:
    return sum(r.cents for r in rows)


def _sample(rows: list[Row], extra=None) -> list[dict[str, Any]]:
    out = []
    for r in sorted(rows, key=lambda r: -abs(r.cents))[:SAMPLE_N]:   # largest first: most informative
        d = {"key": r.key, "amount": r.cents / 100, "line": r.line}
        if r.status: d["status"] = r.status
        if r.day: d["date"] = r.day.isoformat()
        if extra: d.update(extra(r))
        out.append(d)
    return out


def _record(st: State, probe: str, family: str, side: str, desc: str, a_before: int, b_before: int,
            rows: int, samples=None, notes=None) -> Step:
    delta = (tot(st.a_cur) - a_before) - (tot(st.b_cur) - b_before)
    s = Step(f"S{len(st.steps) + 1}", probe, family, side, desc, delta, rows, samples or [], notes or {})
    st.steps.append(s)
    return s


def dedupe(st: State, side: str) -> list[Step]:
    cur = st.a_cur if side == "A" else st.b_cur
    a0, b0 = tot(st.a_cur), tot(st.b_cur)
    seen: dict[tuple[str, int], Row] = {}
    keep, removed = [], []
    amounts_by_key: dict[str, set[int]] = {}
    for r in cur:
        amounts_by_key.setdefault(r.key, set()).add(r.cents)
        if (r.key, r.cents) in seen:
            removed.append(r)
        else:
            seen[(r.key, r.cents)] = r; keep.append(r)
    conflicting = sum(1 for v in amounts_by_key.values() if len(v) > 1)
    if side == "A": st.a_cur = keep
    else: st.b_cur = keep
    return [_record(st, "dedupe", "duplicates", side, f"Exact duplicate records (same key and amount) in source {side}",
                    a0, b0, len(removed), _sample(removed), {"conflicting_keys_left_untouched": conflicting})]


def exclude_status(st: State, side: str, statuses: list[str]) -> list[Step]:
    cur = st.a_cur if side == "A" else st.b_cur
    wanted = {s.strip().lower() for s in statuses}
    present = {r.status for r in cur if r.status}
    unknown = sorted(wanted - present)
    if unknown:
        raise ValueError(f"status value(s) {unknown} not present in source {side}; present: {sorted(present)}")
    a0, b0 = tot(st.a_cur), tot(st.b_cur)
    removed = [r for r in cur if r.status in wanted]
    keep = [r for r in cur if r.status not in wanted]
    if side == "A": st.a_cur = keep
    else: st.b_cur = keep
    return [_record(st, "exclude_status", "status_scope", side,
                    f"Records with status {', '.join(sorted(wanted))} counted in source {side} but not in the other", a0, b0,
                    len(removed), _sample(removed), {"statuses": sorted(wanted)})]


def align_cutoff(st: State) -> list[Step]:
    a_map = {r.key: r for r in st.a_all}
    b_map = {r.key: r for r in st.b_all}
    a_keys = {r.key for r in st.a_cur}
    b_keys = {r.key for r in st.b_cur}
    other = lambda m: (lambda r: {"other_source_date": m[r.key].day.isoformat() if m[r.key].day else None})

    a0, b0 = tot(st.a_cur), tot(st.b_cur)
    moved_out = [r for r in st.a_cur if r.key in b_map and r.key not in b_keys and not st.in_period(b_map[r.key])]
    out_keys = {r.key for r in moved_out}
    st.a_cur = [r for r in st.a_cur if r.key not in out_keys]
    s1 = _record(st, "align_cutoff", "cutoff", "AB", "Same record dated inside the period in A but outside it in B",
                 a0, b0, len(moved_out), _sample(moved_out, other(b_map)), {"direction": "A_in_B_out"})

    a0, b0 = tot(st.a_cur), tot(st.b_cur)
    moved_in = [a_map[r.key] for r in st.b_cur if r.key in a_map and r.key not in a_keys and not st.in_period(a_map[r.key])]
    st.a_cur = st.a_cur + [copy.copy(r) for r in moved_in]
    s2 = _record(st, "align_cutoff", "cutoff", "AB", "Same record dated inside the period in B but outside it in A",
                 a0, b0, len(moved_in), _sample(moved_in, other(b_map)), {"direction": "B_in_A_out"})
    return [s1, s2]


def reconcile_missing(st: State) -> list[Step]:
    a_keys = {r.key for r in st.a_cur}; b_keys = {r.key for r in st.b_cur}
    a0, b0 = tot(st.a_cur), tot(st.b_cur)
    only_a = [r for r in st.a_cur if r.key not in b_keys]
    st.a_cur = [r for r in st.a_cur if r.key in b_keys]
    s1 = _record(st, "reconcile_missing", "missing_records", "A", "Records present in A (in period) with no counterpart in B",
                 a0, b0, len(only_a), _sample(only_a))
    a0, b0 = tot(st.a_cur), tot(st.b_cur)
    only_b = [r for r in st.b_cur if r.key not in a_keys]
    st.a_cur = st.a_cur + [copy.copy(r) for r in only_b]          # bring A to B's view of these records
    s2 = _record(st, "reconcile_missing", "missing_records", "B", "Records present in B (in period) with no counterpart in A",
                 a0, b0, len(only_b), _sample(only_b))
    return [s1, s2]


def _pairs(st: State) -> list[tuple[Row, Row]]:
    b_map: dict[str, Row] = {}
    for r in st.b_cur:
        b_map.setdefault(r.key, r)
    seen, out = set(), []
    for a in st.a_cur:
        if a.key in b_map and a.key not in seen:
            seen.add(a.key); out.append((a, b_map[a.key]))
    return out


def explain_value_gap(st: State, ratio_tolerance: float = 0.01, fit_share: float = 0.95) -> list[Step]:
    pairs = _pairs(st)
    steps: list[Step] = []
    groups: dict[str, list[tuple[Row, Row]]] = {}
    for a, b in pairs:
        groups.setdefault(a.currency or "ALL", []).append((a, b))
    notes: dict[str, Any] = {"matched_pairs": len(pairs)}
    a0, b0 = tot(st.a_cur), tot(st.b_cur)
    scaled_rows, scaled_samples = 0, []
    for g, ps in groups.items():
        ratios = [b.cents / a.cents for a, b in ps if a.cents]
        if not ratios:
            continue
        med = median(ratios)
        if abs(med - 1) <= 0.002:
            continue
        fit = sum(abs(r / med - 1) <= ratio_tolerance for r in ratios) / len(ratios)
        notes[f"ratio_{g}"] = {"median": round(med, 6), "share_fitting": round(fit, 4)}
        if fit >= fit_share:
            r_all = sum(b.cents for _, b in ps) / sum(a.cents for a, _ in ps)
            for a, _ in ps:
                new = int(round(a.cents * r_all))
                if new != a.cents:
                    scaled_rows += 1
                    if len(scaled_samples) < SAMPLE_N:
                        scaled_samples.append({"key": a.key, "before": a.cents / 100, "after": new / 100})
                    a.cents = new
    if scaled_rows:
        steps.append(_record(st, "explain_value_gap", "value_basis", "A",
                             "Constant ratio between the sources' amounts (e.g. tax-inclusive vs net, or FX)",
                             a0, b0, scaled_rows, scaled_samples, notes))
    a0, b0 = tot(st.a_cur), tot(st.b_cur)
    rounded = [(a, b) for a, b in _pairs(st) if a.cents != b.cents and abs(a.cents - b.cents) <= 2]
    for a, b in rounded:
        a.cents = b.cents
    steps.append(_record(st, "explain_value_gap", "value_basis", "A", "Rounding differences of up to 2 cents per record",
                         a0, b0, len(rounded), [{"key": a.key, "amount": a.cents / 100} for a, _ in rounded[:SAMPLE_N]]))
    if not scaled_rows:
        steps[0].notes.update(notes)
    return steps


def inspect_residual(st: State, limit: int = 10) -> dict[str, Any]:
    pairs = [(a, b) for a, b in _pairs(st) if a.cents != b.cents]
    pairs.sort(key=lambda p: -abs(p[1].cents - p[0].cents))
    b_keys = {r.key for r in st.b_cur}; a_keys = {r.key for r in st.a_cur}
    return {"residual": st.residual / 100, "matched_rows_still_differing": len(pairs),
            "top_differences": [{"key": a.key, "a": a.cents / 100, "b": b.cents / 100, "diff": (b.cents - a.cents) / 100}
                                for a, b in pairs[:max(1, min(int(limit), 25))]],
            "unmatched_in_a": sum(r.key not in b_keys for r in st.a_cur),
            "unmatched_in_b": sum(r.key not in a_keys for r in st.b_cur)}
