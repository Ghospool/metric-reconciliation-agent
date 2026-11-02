"""Scripted heuristic policy standing in for Claude so the full loop runs offline.
It maps columns by name, then applies the default probe order. It is a plumbing and engine check,
NOT a measure of how well Claude chooses probes or writes explanations."""
from __future__ import annotations

import json
import re
from typing import Any

from agentic_core import LLMResponse, MockClient, ToolCall

CANNED = {
    "duplicates": ("The same record appears more than once in the source, usually from a double load or a join fan-out.",
                   "Add a uniqueness check on the record key to the load and remove the duplicates at source."),
    "status_scope": ("One source counts records in a status the other excludes, so the two metrics use different scope rules.",
                     "Agree one definition of which statuses count and apply it in both reports."),
    "cutoff": ("Some records fall on different sides of the period boundary in each system, which points to different cut-off or timezone rules.",
               "Align the date field and timezone used to assign records to a period."),
    "missing_records": ("Some records exist in only one system, which suggests a failed or delayed sync, or manual entries.",
                        "Investigate the listed keys with the owner of the missing side and fix the sync."),
    "value_basis": ("Amounts differ by a consistent ratio, consistent with tax-inclusive versus net reporting, or currency conversion.",
                    "Document the basis of each report's amounts and show both on the dashboard."),
}


def _history(messages: list[dict]) -> list[tuple[str, dict, dict]]:
    uses: dict[str, tuple[str, dict]] = {}
    out = []
    for m in messages:
        if not isinstance(m["content"], list):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_use":
                uses[b["id"]] = (b["name"], b["input"])
            elif b.get("type") == "tool_result" and b["tool_use_id"] in uses:
                out.append((*uses[b["tool_use_id"]], json.loads(b["content"])))
    return out


def _pick(cols: dict[str, Any], *pats: str, want: str | None = None) -> str | None:
    for p in pats:
        for c, info in cols.items():
            if re.search(p, c, re.I) and (want is None or info.get("looks_like") == want or want == "any"):
                return c
    return None


def _mapping(side: str, prof: dict) -> dict:
    cols = prof["columns"]
    m = {"source": side,
         "key": _pick(cols, r"(^|_)(id|ref|key)($|_)", r"id|ref") or next(iter(cols)),
         "amount": _pick(cols, r"total|amount|value|revenue|net|gross", want="amount") or _pick(cols, r"total|amount", want="any"),
         "date": _pick(cols, r"date|time|posted", want="date") or _pick(cols, r"date", want="any")}
    st = _pick(cols, r"status|state", want="any")
    cur = _pick(cols, r"curr", want="any")
    if st: m["status"] = st
    if cur: m["currency"] = cur
    return m


def _agent(req: dict[str, Any]) -> LLMResponse:
    hist = _history(req["messages"])
    done = {(h[0], json.dumps(h[1], sort_keys=True)) for h in hist}
    names = [h[0] for h in hist]
    if "profile_sources" not in names:
        return LLMResponse(tool_calls=[ToolCall("profile_sources", {})])
    prof = next(h[2] for h in hist if h[0] == "profile_sources")
    if "declare_mapping" not in names:
        return LLMResponse(tool_calls=[ToolCall("declare_mapping", _mapping("A", prof["A"])),
                                       ToolCall("declare_mapping", _mapping("B", prof["B"]))])
    if "get_totals" not in names:
        return LLMResponse(tool_calls=[ToolCall("get_totals", {})])
    plan: list[tuple[str, dict]] = [("dedupe", {"source": "A"}), ("dedupe", {"source": "B"})]
    for side, other in (("A", "B"), ("B", "A")):
        stcol = next((h[1].get("status") for h in hist if h[0] == "declare_mapping" and h[1]["source"] == side), None)
        if stcol:
            vals = prof[side]["columns"].get(stcol, {}).get("values", {})
            bad = [v for v in vals if re.search(r"cancel|void|refund|test|draft|reject|fail", v, re.I)]
            if bad:
                plan.append(("exclude_status", {"source": side, "statuses": bad}))
    plan += [("align_cutoff", {}), ("reconcile_missing", {}), ("explain_value_gap", {}), ("inspect_residual", {})]
    for name, args in plan:
        if (name, json.dumps(args, sort_keys=True)) not in done:
            return LLMResponse(tool_calls=[ToolCall(name, args)])
    return LLMResponse(tool_calls=[ToolCall("finish", {"notes": "Standard probe sequence completed."})])


def _summary(req: dict[str, Any]) -> LLMResponse:
    br = json.loads(re.search(r"<bridge>\n(.*)\n</bridge>", str(req["messages"][-1]["content"]), re.S)[1])
    findings = [{"step_id": s["id"], "explanation": CANNED[s["family"]][0], "action": CANNED[s["family"]][1]}
                for s in br["steps"] if s["effect"] != 0]
    gap, res = abs(br["gap_cents_B_minus_A"]), abs(br["residual_cents"])
    conf = "high" if gap == 0 or res / gap < 0.01 else "medium" if res / gap < 0.1 else "low"
    caveats = [] if conf == "high" else ["A residual remains that none of the available probes explains; row-level source data is needed."]
    return LLMResponse(tool_calls=[ToolCall("write_summary", {
        "headline": "The gap between the two sources is explained by the steps below.", "findings": findings,
        "confidence": conf, "caveats": caveats})])


def _handler(req: dict[str, Any]) -> LLMResponse:
    return _summary(req) if req.get("force_tool") == "write_summary" else _agent(req)


def offline_client() -> MockClient:
    return MockClient(_handler)
