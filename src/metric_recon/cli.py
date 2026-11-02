"""metric-recon run A.csv B.csv --start 2026-10-01 --end 2026-10-31 --label "October revenue" [--offline]
   metric-recon eval data/scenarios [--offline]"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date

from agentic_core import AnthropicClient, get_settings

from .agent import AgentConfig, ReconAgent
from .evaluate import run_eval
from .offline import offline_client


def main() -> None:
    p = argparse.ArgumentParser(prog="metric-recon")
    p.add_argument("--offline", action="store_true", help="use the scripted mock instead of Claude")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("a_csv"); r.add_argument("b_csv")
    r.add_argument("--start", required=True); r.add_argument("--end", required=True)
    r.add_argument("--label", default="metric"); r.add_argument("--a-name", default="Source A")
    r.add_argument("--b-name", default="Source B"); r.add_argument("--out")
    e = sub.add_parser("eval"); e.add_argument("scenarios_dir")
    a = p.parse_args()
    s = get_settings(); logging.basicConfig(level=s.log_level)
    client = offline_client() if a.offline else AnthropicClient(s)

    async def go() -> None:
        if a.cmd == "run":
            res = await ReconAgent(client, AgentConfig(), s).run(
                a.a_csv, a.b_csv, date.fromisoformat(a.start), date.fromisoformat(a.end), a.label, a.a_name, a.b_name)
            if a.out:
                open(a.out, "w", encoding="utf-8").write(res.report_markdown)
            print(res.report_markdown); print("\n", res.metrics())
        else:
            for k, v in (await run_eval(client, a.scenarios_dir, s)).items():
                print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

    asyncio.run(go())


if __name__ == "__main__":
    main()
