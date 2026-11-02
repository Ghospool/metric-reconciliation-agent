"""Reconciliation agent.

AGENT LOOP (model picks probes, maps columns, decides when to stop; hard budgets)
  -> deterministic BRIDGE  (exact cents; sum of steps + residual == gap, checked)
  -> SUMMARY by the model (explanations and actions only; it may not write numbers)
  -> REPORT rendered by code with verified numbers; unverified numbers in model text are flagged."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any

from agentic_core import LLMClient, Settings, get_settings
from agentic_core.security import AuditLog

from .loader import load_csv
from .money import fmt
from .schemas import Step
from .tools import TOOLS, Executor
from .verify import allowed_numbers, explained_share, unverified_numbers

SUMMARY_TOOL = {
    "name": "write_summary", "description": "Write explanations and actions for each step of the bridge. No numbers.",
    "input_schema": {"type": "object", "required": ["headline", "findings", "confidence", "caveats"], "properties": {
        "headline": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "object", "required": ["step_id", "explanation", "action"],
                     "properties": {"step_id": {"type": "string"}, "explanation": {"type": "string"}, "action": {"type": "string"}}}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "caveats": {"type": "array", "items": {"type": "string"}}}},
}
FAMILY_LABEL = {"duplicates": "Duplicate records", "status_scope": "Status / scope difference", "cutoff": "Period cut-off",
                "missing_records": "Records in one source only", "value_basis": "Amount basis (tax/FX/rounding)"}


def _prompt(name: str) -> str:
    return resources.files("metric_recon.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")


@dataclass
class AgentConfig:
    max_steps: int = 14
    max_tool_calls: int = 30
    max_input_tokens: int = 250_000
    max_idle_turns: int = 2
    residual_review_share: float = 0.05     # above this unexplained share, confidence cannot be "high"


@dataclass
class ReconResult:
    label: str
    report_markdown: str
    steps: list[Step] = field(default_factory=list)
    a_reported: int = 0
    b_reported: int = 0
    residual: int = 0
    explained_share: float = 0.0
    bridge_balances: bool = True
    unverified_numbers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stop_reason: str = "finished"
    turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def gap(self) -> int:
        return self.b_reported - self.a_reported

    def family_totals(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.steps:
            out[s.family] = out.get(s.family, 0) + s.delta_cents
        return out

    def metrics(self) -> dict[str, Any]:
        return {"stop_reason": self.stop_reason, "turns": self.turns, "tool_calls": self.tool_calls,
                "explained_share": round(self.explained_share, 4), "bridge_balances": self.bridge_balances,
                "unverified_numbers": len(self.unverified_numbers), "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens}


class ReconAgent:
    def __init__(self, client: LLMClient, config: AgentConfig | None = None, settings: Settings | None = None) -> None:
        self.client = client
        self.cfg = config or AgentConfig()
        self.s = settings or get_settings()
        self.audit = AuditLog(self.s.audit_log_path)

    async def run(self, a_path: str | Path, b_path: str | Path, start: date, end: date, label: str = "metric",
                  a_name: str = "Source A", b_name: str = "Source B") -> ReconResult:
        u0 = (self.client.total_usage.input_tokens, self.client.total_usage.output_tokens)
        ex = Executor(load_csv(a_path), load_csv(b_path), start, end, self.cfg.max_tool_calls, self.audit)
        messages: list[dict] = [{"role": "user", "content": (
            f"<task>Explain why '{label}' for {start} to {end} differs between {a_name} (source A) and {b_name} "
            f"(source B). Bridge from A to B.</task>")}]
        stop, turns, idle = "finished", 0, 0
        while not ex.finished:
            if turns >= self.cfg.max_steps:
                stop = "max_steps"; break
            if self.client.total_usage.input_tokens - u0[0] > self.cfg.max_input_tokens:
                stop = "token_budget"; break
            if ex.calls >= self.cfg.max_tool_calls:
                stop = "tool_budget"; break
            turns += 1
            resp = await self.client.complete(system=_prompt("research"), messages=messages, model=self.s.smart_model,
                                              max_tokens=1500, tools=TOOLS)
            calls = [(c.name, c.input, c.id or f"toolu_{turns}_{i}") for i, c in enumerate(resp.tool_calls)]
            blocks = ([{"type": "text", "text": resp.text}] if resp.text else []) + \
                     [{"type": "tool_use", "id": i, "name": n, "input": a} for n, a, i in calls]
            messages.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no output)"}]})
            if not calls:
                idle += 1
                if idle > self.cfg.max_idle_turns:
                    stop = "no_progress"; break
                messages.append({"role": "user", "content": "Continue with tools, or call finish."})
                continue
            idle = 0
            results = []
            for n, a, i in calls:                            # strictly sequential: probes are order-dependent
                r = await ex.run(n, a)
                results.append({"type": "tool_result", "tool_use_id": i, "content": json.dumps(r, ensure_ascii=False),
                                "is_error": "error" in r})
            messages.append({"role": "user", "content": results})

        warnings: list[str] = []
        if stop != "finished":
            warnings.append(f"investigation stopped early ({stop}); the bridge may be incomplete")
        for side, iss in ex.issues.items():
            if any(iss.values()):
                warnings.append(f"source {side}: dropped rows during parsing {iss}")
        if ex.state is None:
            warnings.append("no usable mapping was declared; nothing could be reconciled")
            res = ReconResult(label, f"# Reconciliation: {label}\n\nNo reconciliation was possible.\n\n"
                              + "\n".join(f"- {w}" for w in warnings), warnings=warnings, stop_reason=stop,
                              turns=turns, tool_calls=ex.calls)
            return res
        st = ex.state
        steps = st.steps
        balances = sum(s.delta_cents for s in steps) + st.residual == st.gap
        if not balances:                                     # a bug, never an expected state: fail loudly in the report
            warnings.append("INTERNAL ERROR: bridge does not balance; do not use this report")

        summary = await self._summary(label, st, steps, warnings)
        res = ReconResult(label=label, report_markdown="", steps=steps, a_reported=st.a_reported, b_reported=st.b_reported,
                          residual=st.residual, explained_share=explained_share(steps, st.gap), bridge_balances=balances,
                          warnings=warnings, stop_reason=stop, turns=turns, tool_calls=ex.calls)
        res.report_markdown = self._render(res, summary, a_name, b_name, start, end, ex.finish_notes)
        res.input_tokens = self.client.total_usage.input_tokens - u0[0]
        res.output_tokens = self.client.total_usage.output_tokens - u0[1]
        self.audit.write("recon_run", label=label, **res.metrics())
        return res

    async def _summary(self, label, st, steps: list[Step], warnings: list[str]) -> dict:
        bridge = {"label": label, "gap_cents_B_minus_A": st.gap, "residual_cents": st.residual,
                  "steps": [{"id": s.id, "family": s.family, "side": s.side, "description": s.description,
                             "effect": s.delta_cents / 100, "records": s.rows, "samples": s.samples[:3], "notes": s.notes}
                            for s in steps if not s.no_effect]}
        try:
            r = await self.client.complete(
                system=_prompt("summary"), model=self.s.smart_model, max_tokens=2000, tools=[SUMMARY_TOOL],
                force_tool="write_summary",
                messages=[{"role": "user", "content": f"<bridge>\n{json.dumps(bridge, ensure_ascii=False)}\n</bridge>"}])
            for c in r.tool_calls:
                if c.name == "write_summary":
                    return c.input
        except Exception as e:  # noqa: BLE001
            warnings.append(f"summary step failed ({type(e).__name__}); report contains the numeric bridge only")
            return {}
        warnings.append("summary step returned nothing; report contains the numeric bridge only")
        return {}

    def _render(self, res: ReconResult, summ: dict, a_name: str, b_name: str, start: date, end: date, notes: str) -> str:
        gap = res.gap
        active = [s for s in res.steps if not s.no_effect]
        zero = [s for s in res.steps if s.no_effect]
        L = [f"# Reconciliation: {res.label}", "", f"Period {start} to {end}. Bridging **{a_name}** (A) to **{b_name}** (B).", ""]
        head = str(summ.get("headline", "")).strip()
        allowed = allowed_numbers(res.steps, res.a_reported, res.b_reported, res.residual, gap)
        bad: list[str] = []

        def checked(t: str) -> str:
            b = unverified_numbers(t, allowed)
            bad.extend(b)
            return t + (" [contains unverified numbers]" if b else "")

        if head:
            L += [checked(head), ""]
        L += ["| | Amount |", "|---|---:|", f"| A reported | {fmt(res.a_reported)} |", f"| B reported | {fmt(res.b_reported)} |",
              f"| **Gap (B - A)** | **{fmt(gap, True)}** |", ""]
        if gap == 0 and not active:
            L += ["The two sources agree for this period. No causes were identified because there is nothing to explain.", ""]
        else:
            L += ["## Bridge from A to B", "", "| Step | Cause | Side | Records | Effect | % of gap |", "|---|---|---|---:|---:|---:|"]
            for s in active:
                pct = f"{s.delta_cents / gap * 100:.1f}%" if gap else "-"
                L.append(f"| {s.id} | {FAMILY_LABEL.get(s.family, s.family)}: {s.description} | {s.side} | {s.rows} | {fmt(s.delta_cents, True)} | {pct} |")
            rp = f"{res.residual / gap * 100:.1f}%" if gap else "-"
            L += [f"| - | **Unexplained residual** | - | - | {fmt(res.residual, True)} | {rp} |",
                  f"| | **Total = gap** | | | **{fmt(gap, True)}** | 100% |", "",
                  f"Explained: **{res.explained_share * 100:.1f}%** of the gap. The bridge is order-dependent: steps were applied in the order shown.", ""]
        valid = {s.id for s in active}
        fnds = [f for f in summ.get("findings", []) if f.get("step_id") in valid]
        if len(fnds) != len(summ.get("findings", [])):
            res.warnings.append("summary referred to steps that do not exist; those findings were dropped")
        if fnds:
            L += ["## Findings and recommended actions", ""]
            for f in fnds:
                L += [f"**{f['step_id']}.** {checked(str(f.get('explanation', '')))}", "",
                      f"*Action:* {checked(str(f.get('action', '')))}", ""]
            missing = valid - {f["step_id"] for f in fnds}
            if missing:
                res.warnings.append(f"no explanation was written for step(s) {sorted(missing)}")
        for s in active:
            if s.samples:
                L += [f"### Evidence for {s.id}", "", "| Key | Amount | Detail |", "|---|---:|---|"]
                for r in s.samples[:5]:
                    detail = ", ".join(f"{k}={v}" for k, v in r.items() if k not in ("key", "amount", "line"))
                    L.append(f"| {r.get('key')} | {r.get('amount', r.get('after', ''))} | {detail} (line {r.get('line', '-')}) |")
                L.append("")
        if zero:
            L += ["## Checked, no effect", "", ", ".join(sorted({f"{s.probe} ({s.side})" for s in zero})), ""]
        conf = summ.get("confidence", "low")
        if conf == "high" and gap and abs(res.residual) / abs(gap) > self.cfg.residual_review_share:
            conf = "medium"
            res.warnings.append("confidence lowered from high: the unexplained residual exceeds the review threshold")
        if gap and abs(res.residual) / abs(gap) > 0.25:
            conf = "low"
        L += [f"## Confidence: {conf}", ""]
        for c in summ.get("caveats", []):
            L += [f"- {checked(str(c))}"]
        if notes:
            L += ["", f"Agent notes: {checked(notes)}"]
        res.unverified_numbers = bad
        if bad:
            res.warnings.append(f"{len(bad)} number(s) in model-written text could not be traced to the bridge: {sorted(set(bad))[:5]}")
        if res.warnings:
            L += ["", "## Reviewer notes", ""] + [f"- {w}" for w in res.warnings]
        return "\n".join(L)
