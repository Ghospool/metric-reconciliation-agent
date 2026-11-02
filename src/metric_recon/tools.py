"""Tool surface for the agent. Read-only on the files; state-mutating probes run strictly in call order."""
from __future__ import annotations

from datetime import date
from typing import Any

from agentic_core.security import AuditLog

from . import probes
from .loader import normalise, profile
from .schemas import Mapping, Step

SRC = {"type": "string", "enum": ["A", "B"]}
TOOLS = [
    {"name": "profile_sources", "description": "Column profile of both sources: row counts, types, value ranges, and the values of low-cardinality columns such as status.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "declare_mapping", "description": "Declare which columns hold the record key, amount, date, and optionally status and currency for one source. A date column is required.",
     "input_schema": {"type": "object", "required": ["source", "key", "amount", "date"], "properties": {
         "source": SRC, "key": {"type": "string"}, "amount": {"type": "string"}, "date": {"type": "string"},
         "status": {"type": "string"}, "currency": {"type": "string"}}}},
    {"name": "get_totals", "description": "Reported totals of A and B for the period using the declared mappings, and the gap to explain.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "dedupe", "description": "Remove exact duplicate records (same key and amount) from one source and record the effect on the bridge.",
     "input_schema": {"type": "object", "required": ["source"], "properties": {"source": SRC}}},
    {"name": "exclude_status", "description": "Remove records with the given status values from one source (for example cancelled or test orders that the other source does not count).",
     "input_schema": {"type": "object", "required": ["source", "statuses"], "properties": {
         "source": SRC, "statuses": {"type": "array", "items": {"type": "string"}, "minItems": 1}}}},
    {"name": "align_cutoff", "description": "Handle records present in both sources but dated inside the period in one and outside it in the other (period cut-off or timezone differences).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "reconcile_missing", "description": "Handle records that exist in only one source after the earlier steps.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "explain_value_gap", "description": "On matched records, test for a constant amount ratio (tax-inclusive vs net, FX) and for rounding differences.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "inspect_residual", "description": "Read-only: show what still differs after the steps so far (largest differences, unmatched counts).",
     "input_schema": {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 25}}}},
    {"name": "finish", "description": "Call when the gap is explained or nothing more can be explained with the available probes.",
     "input_schema": {"type": "object", "required": ["notes"], "properties": {"notes": {"type": "string"}}}},
]


class Executor:
    def __init__(self, a_rows: list[dict], b_rows: list[dict], start: date, end: date, max_calls: int,
                 audit: AuditLog | None = None) -> None:
        self.raw = {"A": a_rows, "B": b_rows}
        self.start, self.end, self.max_calls = start, end, max_calls
        self.audit = audit or AuditLog(None)
        self.maps: dict[str, Mapping] = {}
        self.norm: dict[str, list] = {}
        self.issues: dict[str, dict[str, int]] = {}
        self.state: probes.State | None = None
        self.calls = 0
        self.finished = False
        self.finish_notes = ""

    # ----- dispatch -----
    async def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        if self.calls > self.max_calls:
            res: dict[str, Any] = {"error": "tool-call budget exhausted; call finish"}
        else:
            try:
                fn = getattr(self, f"_t_{name}", None)
                res = fn(**args) if fn else {"error": f"unknown tool {name}"}
            except (ValueError, TypeError, KeyError) as e:       # bad arguments are fed back to the agent
                res = {"error": f"{type(e).__name__}: {e}"[:300]}
        self.audit.write("tool_call", tool=name, args={k: str(v)[:60] for k, v in args.items()}, ok="error" not in res)
        return res

    def _need_state(self) -> probes.State:
        if self.state is None:
            if set(self.maps) != {"A", "B"}:
                raise ValueError("declare_mapping for both sources A and B first")
            self.state = probes.State(self.norm["A"], self.norm["B"], self.start, self.end)
        return self.state

    @staticmethod
    def _steps(steps: list[Step], st: probes.State) -> dict[str, Any]:
        return {"steps": [{"id": s.id, "family": s.family, "side": s.side, "description": s.description,
                           "effect": s.delta_cents / 100, "records": s.rows, "samples": s.samples, "notes": s.notes,
                           "no_effect": s.no_effect} for s in steps],
                "remaining_gap": st.residual / 100}

    # ----- tools -----
    def _t_profile_sources(self) -> dict:
        return {"period": [self.start.isoformat(), self.end.isoformat()],
                "A": profile(self.raw["A"]), "B": profile(self.raw["B"])}

    def _t_declare_mapping(self, source: str, key: str, amount: str, date: str, status: str | None = None,
                           currency: str | None = None) -> dict:
        if source not in ("A", "B"):
            raise ValueError("source must be A or B")
        m = Mapping(key, amount, date, status, currency)
        rows, issues = normalise(self.raw[source], m)
        self.maps[source], self.norm[source], self.issues[source] = m, rows, issues
        self.state = None                                    # any earlier probe results are void
        return {"ok": True, "usable_rows": len(rows), "dropped": issues,
                "note": "previous probe results were reset" if self.state is None and self.calls > 2 else ""}

    def _t_get_totals(self) -> dict:
        st = self._need_state()
        return {"A_reported": st.a_reported / 100, "B_reported": st.b_reported / 100, "gap_B_minus_A": st.gap / 100,
                "rows_in_period": {"A": len(st.a_cur), "B": len(st.b_cur)}}

    def _t_dedupe(self, source: str) -> dict:
        st = self._need_state(); return self._steps(probes.dedupe(st, _side(source)), st)

    def _t_exclude_status(self, source: str, statuses: list[str]) -> dict:
        st = self._need_state()
        if not self.maps[_side(source)].status:
            raise ValueError(f"no status column declared for source {source}")
        return self._steps(probes.exclude_status(st, _side(source), list(statuses)), st)

    def _t_align_cutoff(self) -> dict:
        st = self._need_state(); return self._steps(probes.align_cutoff(st), st)

    def _t_reconcile_missing(self) -> dict:
        st = self._need_state(); return self._steps(probes.reconcile_missing(st), st)

    def _t_explain_value_gap(self) -> dict:
        st = self._need_state(); return self._steps(probes.explain_value_gap(st), st)

    def _t_inspect_residual(self, limit: int = 10) -> dict:
        return probes.inspect_residual(self._need_state(), limit)

    def _t_finish(self, notes: str = "") -> dict:
        self._need_state()
        self.finished, self.finish_notes = True, notes
        return {"ok": True}


def _side(s: str) -> str:
    if s not in ("A", "B"):
        raise ValueError("source must be A or B")
    return s
