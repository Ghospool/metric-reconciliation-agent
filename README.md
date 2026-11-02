# Metric Reconciliation Agent

 An autonomous agent plus a deterministic engine that explains why two reports disagree. Part of my [Claude agentic workflows portfolio](https://github.com/Ghospool/claude-agentic-workflows_practice); this repo is self-contained and includes the small shared core it uses in `src/agentic_core`.

*"Why does the dashboard say 845,645.85 when finance says 680,727.65?"* Every analyst has been asked this. This project answers it with an **auditable bridge** from one number to the other, exact to the cent, with every cause quantified, evidenced, and explained.

```
two extracts (CSV) + period
  -> AGENT LOOP: profile -> declare column mapping -> probes in order -> inspect residual -> finish
       dedupe -> exclude_status -> align_cutoff -> reconcile_missing -> explain_value_gap
  -> BRIDGE (integer cents): sum(step effects) + residual == gap, verified on every run
  -> SUMMARY: the model writes explanations and actions, never numbers
  -> REPORT: rendered by code; any number in model text not found in the bridge is flagged
```

Sample output (scenario with five planted causes): the gap of -164,918.20 decomposes into duplicates -11,561.63, cancelled orders -12,322.77, period cut-off -3,067.05, records in one system only -2,469.92, a constant 20% amount basis (tax-inclusive vs net) -135,497.04, and an unexplained residual of +0.21.

**What is new compared with row-comparison tools.** Libraries such as [DataComPy](https://github.com/capitalone/datacompy) tell you *which rows* differ. This tells you *why the totals differ and how much each cause contributes*, as a waterfall a finance lead can sign off. I searched GitHub for an open-source equivalent and found row-level comparers, vendor material and write-ups of the idea, but no project producing a dollar-exact causal bridge; that search was brief, so treat it as "I did not find one", not "none exists".

**Design decisions**
- **The model never computes.** Probes are deterministic code over integer cents. The model chooses columns, order and stopping point, and writes explanations. Its text may not contain numbers; unverifiable numbers are flagged in the report.
- **Honest residuals.** If no probe explains the remainder, the report says so and caps confidence. A test with random value noise proves the engine does not force-fit a cause.
- **The bridge always balances.** The invariant `sum(effects) + residual == gap` is checked per run (and property-tested on random data); a failure marks the report unusable.
- **Order-dependent by nature.** A waterfall depends on step order. The order is shown, and the default order puts structural causes (duplicates, scope, cut-off, missing records) before amount-basis causes.
- **Evidence, not assertion.** Each step lists the largest affected records. Only mapped columns appear in samples, so unrelated personal data never reaches the model or the report; profiles show values only for low-cardinality columns and mask PII-like text.
- **Fail-soft.** Bad tool arguments return errors the agent can fix; a failed summary still yields the numeric bridge; budgets (turns, tool calls, tokens) end the run with a visible warning.

**Run**
```bash
pip install -e .                                          # Python 3.10+; offline mode needs nothing else
python -m unittest discover -s tests                      # 26 offline tests (set PYTHONPATH=src if not installed)
python data/generate_scenarios.py                         # regenerate the 5 ground-truth scenarios
metric-recon --offline eval data/scenarios
metric-recon --offline run data/scenarios/s3_everything/a.csv data/scenarios/s3_everything/b.csv \
    --start 2026-10-01 --end 2026-10-31 --label "October 2026 revenue" --a-name "BI dashboard" --b-name "Finance ledger"
cp .env.example .env                                      # add ANTHROPIC_API_KEY, then drop --offline
```
Use your own data: two CSVs, each with a record key, an amount and a date column (status and currency optional).

**Evaluation** (`data/scenarios`, planted causes with independently computed ground truth): cause recall, spurious-cause share, residual accuracy, balance of the bridge, unverified numbers, turns, tool calls and tokens. Scenarios: structural causes only; tax basis plus missing records; everything combined; no gap at all (must invent nothing); and unexplained value noise (must stay unexplained).

**What the offline score does and does not show.** It validates the engine, the bridge arithmetic, the safeguards and the report. The offline "model" is a scripted policy, so it says nothing about how well Claude maps columns, picks probes, or writes explanations. That needs a live run with your API key, ideally on real extracts.

**Known limits:** one metric (a sum of amounts) over a period; matching is by a single key; no fuzzy matching of keys; the five probes cover common causes but not, for example, currency conversion with changing rates, many-to-many joins or grain differences (counts of users vs orders); the bridge is order-dependent; CSV only (a Power BI or SQL source adapter is the natural next step).
