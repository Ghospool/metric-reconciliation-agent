"""Score the agent against planted ground truth: recall of causes, spurious causes, residual honesty, fidelity."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .agent import AgentConfig, ReconAgent


def _tol(truth: int, gap: int) -> int:
    return max(200, int(abs(truth) * 0.01))          # 1% of the cause, or two currency units


async def run_eval(client, scen_dir: str | Path, settings=None) -> dict:
    out = {"scenarios": 0, "passed": 0, "causes_planted": 0, "causes_found": 0, "spurious_share_max": 0.0,
           "residual_errors": 0, "unverified_numbers": 0, "unbalanced_bridges": 0, "turns": 0, "tool_calls": 0,
           "input_tokens": 0, "output_tokens": 0}
    detail = []
    for d in sorted(p for p in Path(scen_dir).iterdir() if (p / "truth.json").exists()):
        t = json.loads((d / "truth.json").read_text())
        res = await ReconAgent(client, AgentConfig(), settings).run(
            d / "a.csv", d / "b.csv", date.fromisoformat(t["period"][0]), date.fromisoformat(t["period"][1]), t["label"])
        found = res.family_totals(); gap = t["gap"]
        ok_causes = [f for f, v in t["families"].items() if abs(found.get(f, 0) - v) <= _tol(v, gap)]
        spurious = sum(abs(v) for f, v in found.items() if f not in t["families"])
        spurious_share = spurious / abs(gap) if gap else (1.0 if spurious else 0.0)
        resid_ok = abs(res.residual - t["unexplained"]) <= max(500, int(abs(gap) * 0.01))
        passed = (len(ok_causes) == len(t["families"]) and spurious_share <= 0.01 and resid_ok
                  and not res.unverified_numbers and res.bridge_balances)
        out["scenarios"] += 1; out["passed"] += passed
        out["causes_planted"] += len(t["families"]); out["causes_found"] += len(ok_causes)
        out["spurious_share_max"] = max(out["spurious_share_max"], spurious_share)
        out["residual_errors"] += not resid_ok; out["unverified_numbers"] += len(res.unverified_numbers)
        out["unbalanced_bridges"] += not res.bridge_balances
        out["turns"] += res.turns; out["tool_calls"] += res.tool_calls
        out["input_tokens"] += res.input_tokens; out["output_tokens"] += res.output_tokens
        detail.append(f"{d.name}: {'PASS' if passed else 'FAIL'} causes {len(ok_causes)}/{len(t['families'])} "
                      f"explained {res.explained_share * 100:.1f}% spurious {spurious_share * 100:.2f}%")
    out["cause_recall"] = out["causes_found"] / max(1, out["causes_planted"])
    out["scenario_pass_rate"] = out["passed"] / max(1, out["scenarios"])
    out["detail"] = " | ".join(detail)
    return out
