import json
import random
import unittest
from datetime import date
from pathlib import Path

from agentic_core import LLMResponse, MockClient, ToolCall
from agentic_core.security import AuditLog
from metric_recon import AgentConfig, ReconAgent
from metric_recon.evaluate import run_eval
from metric_recon.loader import load_csv, normalise, profile
from metric_recon.money import fmt, parse_cents, parse_day
from metric_recon.offline import _handler, offline_client
from metric_recon.probes import State, align_cutoff, dedupe, exclude_status, explain_value_gap, reconcile_missing
from metric_recon.schemas import Mapping
from metric_recon.tools import Executor
from metric_recon.verify import allowed_numbers, unverified_numbers

DATA = Path(__file__).resolve().parents[1] / "data" / "scenarios"
P0, P1 = date(2026, 10, 1), date(2026, 10, 31)
M = Mapping("id", "amt", "d", "st")


def rows(*r):
    return [dict(id=i, d=d, amt=a, st=s) for i, d, a, s in r]


def state(a, b):
    na, _ = normalise(a, M); nb, _ = normalise(b, M)
    return State(na, nb, P0, P1)


class MoneyTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_cents("$1,234.567"), 123457)
        self.assertEqual(parse_cents("(12.50)"), -1250)
        self.assertEqual(parse_cents("-0.05"), -5)
        for bad in ("", "abc", "1.2.3"):
            self.assertIsNone(parse_cents(bad))
        self.assertEqual(parse_day("2026-10-31 23:59:00"), date(2026, 10, 31))
        self.assertIsNone(parse_day("31/10/2026"))
        self.assertEqual(fmt(-123456, True), "-1,234.56")

    def test_normalise_counts_bad_rows_and_rejects_unknown_columns(self):
        r, issues = normalise(rows(("1", "2026-10-01", "5", "x"), ("", "2026-10-01", "5", "x"),
                                   ("3", "2026-10-01", "n/a", "x"), ("4", "bad", "1", "x")), M)
        self.assertEqual((len(r), issues), (2, {"blank_key": 1, "bad_amount": 1, "bad_date": 1}))
        with self.assertRaises(ValueError):
            normalise(rows(("1", "2026-10-01", "5", "x")), Mapping("nope", "amt", "d"))


class ProbeTests(unittest.TestCase):
    def test_dedupe_only_exact_duplicates(self):
        a = rows(("1", "2026-10-02", "10", "ok"), ("1", "2026-10-02", "10", "ok"), ("2", "2026-10-03", "5", "ok"),
                 ("2", "2026-10-03", "7", "ok"))
        st = state(a, rows(("1", "2026-10-02", "10", "ok")))
        s = dedupe(st, "A")[0]
        self.assertEqual((s.delta_cents, s.rows, s.notes["conflicting_keys_left_untouched"]), (-1000, 1, 1))

    def test_dedupe_in_b_has_positive_contribution(self):
        st = state(rows(("1", "2026-10-02", "10", "ok")),
                   rows(("1", "2026-10-02", "10", "ok"), ("1", "2026-10-02", "10", "ok")))
        s = dedupe(st, "B")[0]
        self.assertEqual(s.delta_cents, 1000)          # B overstated by a duplicate: the bridge must add it back
        self.assertEqual(sum(x.delta_cents for x in [s]) + st.residual, st.gap)

    def test_exclude_status_validates_values(self):
        st = state(rows(("1", "2026-10-02", "10", "ok"), ("2", "2026-10-02", "4", "Cancelled")), [])
        with self.assertRaises(ValueError):
            exclude_status(st, "A", ["void"])
        self.assertEqual(exclude_status(st, "A", [" cancelled "])[0].delta_cents, -400)

    def test_cutoff_both_directions(self):
        a = rows(("1", "2026-10-31", "10", ""), ("2", "2026-09-30", "6", ""))
        b = rows(("1", "2026-11-02", "10", ""), ("2", "2026-10-02", "6", ""))
        st = state(a, b)
        out, inn = align_cutoff(st)
        self.assertEqual((out.delta_cents, out.rows, inn.delta_cents, inn.rows), (-1000, 1, 600, 1))
        self.assertEqual(st.residual, 0)

    def test_missing_both_sides(self):
        st = state(rows(("1", "2026-10-02", "10", ""), ("2", "2026-10-02", "3", "")),
                   rows(("1", "2026-10-02", "10", ""), ("9", "2026-10-02", "8", "")))
        a, b = reconcile_missing(st)
        self.assertEqual((a.delta_cents, b.delta_cents, st.residual), (-300, 800, 0))

    def test_constant_ratio_and_rounding(self):
        a = rows(*[(str(i), "2026-10-02", f"{round(i * 1.2, 2):.2f}", "") for i in range(1, 60)])
        b = rows(*[(str(i), "2026-10-02", f"{i:.2f}", "") for i in range(1, 60)])
        st = state(a, b)
        steps = explain_value_gap(st)
        self.assertEqual(st.residual, 0)
        self.assertEqual(sum(s.delta_cents for s in steps) + st.residual, st.gap)

    def test_no_constant_ratio_leaves_residual(self):
        rnd = random.Random(1)
        a = rows(*[(str(i), "2026-10-02", f"{rnd.uniform(10, 100):.2f}", "") for i in range(100)])
        b = rows(*[(r["id"], "2026-10-02", f"{float(r['amt']) * rnd.uniform(0.7, 1.4):.2f}", "") for r in a])
        st = state(a, b)
        explain_value_gap(st)
        self.assertNotEqual(st.residual, 0)               # it must NOT force-fit an explanation

    def test_bridge_always_balances_on_random_data(self):
        for seed in range(25):
            rnd = random.Random(seed)
            days = ["2026-09-30", "2026-10-05", "2026-10-31", "2026-11-02"]
            def mk(pref):
                return rows(*[(f"{pref}{rnd.randrange(60)}", rnd.choice(days), f"{rnd.randint(1, 5000) / 100:.2f}",
                               rnd.choice(["ok", "void"])) for _ in range(80)])
            st = state(mk(""), mk(""))
            steps = dedupe(st, "A") + dedupe(st, "B") + exclude_status(st, "A", ["void"]) + align_cutoff(st) \
                + reconcile_missing(st) + explain_value_gap(st)
            self.assertEqual(sum(s.delta_cents for s in steps) + st.residual, st.gap, seed)


class ToolTests(unittest.IsolatedAsyncioTestCase):
    def make(self, a, b, cap=40):
        ex = Executor(a, b, P0, P1, cap, AuditLog(None))
        return ex

    async def test_probe_needs_mapping_and_validates(self):
        ex = self.make(rows(("1", "2026-10-02", "10", "ok")), rows(("1", "2026-10-02", "10", "ok")))
        self.assertIn("declare_mapping", (await ex.run("dedupe", {"source": "A"}))["error"])
        self.assertIn("unknown column", (await ex.run("declare_mapping", {"source": "A", "key": "zz", "amount": "amt", "date": "d"}))["error"])
        self.assertIn("error", await ex.run("declare_mapping", {"source": "C", "key": "id", "amount": "amt", "date": "d"}))
        self.assertIn("error", await ex.run("nonexistent", {}))
        self.assertIn("error", await ex.run("dedupe", {}))

    async def test_budget(self):
        ex = self.make(rows(("1", "2026-10-02", "10", "ok")), rows(("1", "2026-10-02", "10", "ok")), cap=2)
        await ex.run("profile_sources", {}); await ex.run("profile_sources", {})
        self.assertIn("budget", (await ex.run("profile_sources", {}))["error"])

    async def test_profile_hides_free_text_and_masks_pii(self):
        a = [dict(id=f"K{i}", d="2026-10-02", amt=str(i + 10), st="ok", name=f"Person Number {i}",
                  contact="x%d@corp.com" % i) for i in range(40)]
        ex = self.make(a, a)
        out = json.dumps(await ex.run("profile_sources", {}))
        self.assertNotIn("Person Number 7", out)
        self.assertNotIn("x7@corp.com", out)

    async def test_samples_exclude_unmapped_columns(self):
        a = [dict(id="1", d="2026-10-02", amt="10", st="ok", secret="TOP-SECRET-PII")] * 2
        b = [dict(id="1", d="2026-10-02", amt="10", st="ok", secret="TOP-SECRET-PII")]
        ex = self.make(a, b)
        for s in "AB":
            await ex.run("declare_mapping", {"source": s, "key": "id", "amount": "amt", "date": "d", "status": "st"})
        self.assertNotIn("TOP-SECRET", json.dumps(await ex.run("dedupe", {"source": "A"})))


class AgentTests(unittest.IsolatedAsyncioTestCase):
    def agent(self, client=None, **cfg):
        return ReconAgent(client or offline_client(), AgentConfig(**cfg))

    async def go(self, name, agent=None):
        d = DATA / name
        t = json.loads((d / "truth.json").read_text())
        return t, await (agent or self.agent()).run(d / "a.csv", d / "b.csv", date.fromisoformat(t["period"][0]),
                                                    date.fromisoformat(t["period"][1]), t["label"])

    async def test_recovers_planted_causes_exactly(self):
        t, res = await self.go("s3_everything")
        got = res.family_totals()
        for fam, v in t["families"].items():
            self.assertLessEqual(abs(got[fam] - v), max(200, abs(v) // 100), fam)
        self.assertTrue(res.bridge_balances)
        self.assertGreater(res.explained_share, 0.999)
        self.assertEqual(res.unverified_numbers, [])

    async def test_no_gap_invents_nothing(self):
        t, res = await self.go("s4_no_gap")
        self.assertEqual((res.gap, len(res.steps) and sum(s.delta_cents for s in res.steps)), (0, 0))
        self.assertIn("nothing to explain", res.report_markdown)

    async def test_unexplained_noise_stays_unexplained(self):
        t, res = await self.go("s5_unexplained")
        self.assertAlmostEqual(res.residual, t["unexplained"], delta=500)
        self.assertLess(res.explained_share, 0.99)
        self.assertIn("Unexplained residual", res.report_markdown)
        self.assertNotIn("Confidence: high", res.report_markdown)

    async def test_fabricated_numbers_in_narrative_are_flagged(self):
        def h(req):
            r = _handler(req)
            if req.get("force_tool") == "write_summary":
                r.tool_calls[0].input["findings"][0]["explanation"] += " This cost the business 9,999,999.00 or 87.3%."
            return r
        t, res = await self.go("s1_structural", self.agent(MockClient(h)))
        self.assertGreaterEqual(len(res.unverified_numbers), 2)
        self.assertIn("contains unverified numbers", res.report_markdown)

    async def test_hallucinated_step_ids_dropped(self):
        def h(req):
            r = _handler(req)
            if req.get("force_tool") == "write_summary":
                r.tool_calls[0].input["findings"].append({"step_id": "S99", "explanation": "x", "action": "y"})
            return r
        t, res = await self.go("s1_structural", self.agent(MockClient(h)))
        self.assertNotIn("S99", res.report_markdown.split("## Reviewer notes")[0])
        self.assertTrue(any("do not exist" in w for w in res.warnings))

    async def test_overconfident_summary_is_downgraded(self):
        def h(req):
            r = _handler(req)
            if req.get("force_tool") == "write_summary":
                r.tool_calls[0].input["confidence"] = "high"
            return r
        t, res = await self.go("s5_unexplained", self.agent(MockClient(h)))
        self.assertNotIn("Confidence: high", res.report_markdown)

    async def test_budget_stop_still_yields_honest_partial_bridge(self):
        t, res = await self.go("s3_everything", self.agent(max_steps=5))
        self.assertEqual(res.stop_reason, "max_steps")
        self.assertTrue(any("stopped early" in w for w in res.warnings))
        self.assertTrue(res.bridge_balances)
        self.assertLess(res.explained_share, 0.9)

    async def test_idle_model_stops(self):
        t, res = await self.go("s1_structural", self.agent(MockClient(lambda r: LLMResponse(text="hmm"))))
        self.assertEqual(res.stop_reason, "no_progress")
        self.assertIn("No reconciliation was possible", res.report_markdown)

    async def test_summary_failure_degrades_to_numeric_bridge(self):
        def h(req):
            if req.get("force_tool") == "write_summary":
                raise RuntimeError("boom")
            return _handler(req)
        t, res = await self.go("s1_structural", self.agent(MockClient(h)))
        self.assertIn("## Bridge from A to B", res.report_markdown)
        self.assertTrue(any("summary step failed" in w for w in res.warnings))

    async def test_bad_tool_arguments_are_recoverable(self):
        n = {"i": 0}
        def h(req):
            if req.get("force_tool"): return _handler(req)
            n["i"] += 1
            if n["i"] == 1:
                return LLMResponse(tool_calls=[ToolCall("exclude_status", {"source": "A", "statuses": ["nope"]})])
            return _handler(req)
        t, res = await self.go("s1_structural", self.agent(MockClient(h)))
        self.assertEqual(res.stop_reason, "finished")
        self.assertTrue(res.bridge_balances)

    async def test_golden_eval(self):
        r = await run_eval(offline_client(), DATA)
        self.assertEqual((r["scenario_pass_rate"], r["cause_recall"], r["unbalanced_bridges"]), (1.0, 1.0, 0))
        self.assertLessEqual(r["spurious_share_max"], 0.01)


class VerifyTests(unittest.TestCase):
    def test_number_check(self):
        from metric_recon.schemas import Step
        steps = [Step("S1", "dedupe", "duplicates", "A", "d", -12345, 7)]
        ok = allowed_numbers(steps, 100000, 87655, 0, -12345)
        self.assertEqual(unverified_numbers("S1 removed 7 records worth 123.45, dated 2026-10-31.", ok), [])
        self.assertEqual(unverified_numbers("It cost 5,000.00 (12%).", ok), ["5,000.00", "12%"])


if __name__ == "__main__":
    unittest.main()
