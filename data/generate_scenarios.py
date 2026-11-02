"""Synthetic reconciliation scenarios with ground truth.

Source A = a BI dashboard extract; source B = the finance ledger. Both claim 'October 2026 revenue'.
Known causes are planted with exact amounts; truth.json records each cause's contribution to the A -> B
bridge (computed here independently of the library) and any deliberately unexplained residual."""
import csv
import json
import pathlib
import random
from datetime import date, timedelta

OCT1, OCT31 = date(2026, 10, 1), date(2026, 10, 31)
SCENARIOS = {
    "s1_structural":  dict(dups=25, cancelled=30, cutoff=12, prior=6, missing_b=0, missing_a=0, scale=False, noise=0),
    "s2_tax_missing": dict(dups=0, cancelled=0, cutoff=0, prior=0, missing_b=15, missing_a=8, scale=True, noise=0),
    "s3_everything":  dict(dups=20, cancelled=25, cutoff=10, prior=5, missing_b=12, missing_a=6, scale=True, noise=0),
    "s4_no_gap":      dict(dups=0, cancelled=0, cutoff=0, prior=0, missing_b=0, missing_a=0, scale=False, noise=0),
    "s5_unexplained": dict(dups=10, cancelled=0, cutoff=0, prior=0, missing_b=0, missing_a=0, scale=False, noise=45),
}


def a_amount(net, scale):
    return (net * 12 + 5) // 10 if scale else net            # +20% tax, rounded half up, integer cents


def build(name, cfg, seed):
    rnd = random.Random(seed)
    n = 1500
    base = [dict(id=f"ORD-{i:05d}", net=rnd.randint(2000, 90000), day=OCT1 + timedelta(days=rnd.randrange(0, 30)),
                 email=f"user{i}@example.com", region=rnd.choice(["EMEA", "AMER", "APAC"])) for i in range(1, n + 1)]
    idx = rnd.sample(range(n), cfg["cutoff"] + cfg["prior"] + cfg["dups"] + cfg["noise"])
    cut_i, prior_i = idx[:cfg["cutoff"]], idx[cfg["cutoff"]:cfg["cutoff"] + cfg["prior"]]
    dup_i = idx[cfg["cutoff"] + cfg["prior"]:cfg["cutoff"] + cfg["prior"] + cfg["dups"]]
    noise_i = idx[cfg["cutoff"] + cfg["prior"] + cfg["dups"]:]
    A, B = [], []
    amt = {}
    for i, o in enumerate(base):
        da, db, nb = o["day"], o["day"], o["net"]
        if i in cut_i: da, db = date(2026, 10, 31), date(2026, 11, 2)
        if i in prior_i: da, db = date(2026, 9, 30), date(2026, 10, 2)
        a = a_amount(o["net"], cfg["scale"])
        noise = 0
        if i in noise_i:
            noise = rnd.choice([-1, 1]) * rnd.randint(500, 6000); nb = o["net"] + noise
        amt[i] = (a, nb, noise)
        A.append(dict(order_id=o["id"], order_date=da.isoformat(), customer_email=o["email"], region=o["region"],
                      order_total=f"{a / 100:.2f}", order_status="completed"))
        B.append(dict(txn_ref=o["id"], posting_date=db.isoformat(), amount=f"{nb / 100:.2f}", ledger_status="posted"))
    for i in dup_i:
        A.append(dict(A[i]))
    canc = []
    for k in range(cfg["cancelled"]):
        net = rnd.randint(2000, 90000); a = a_amount(net, cfg["scale"])
        canc.append(a)
        A.append(dict(order_id=f"ORD-C{k:04d}", order_date=(OCT1 + timedelta(days=rnd.randrange(0, 30))).isoformat(),
                      customer_email=f"c{k}@example.com", region="EMEA", order_total=f"{a / 100:.2f}", order_status="cancelled"))
    miss_b = []
    for k in range(cfg["missing_b"]):
        net = rnd.randint(2000, 90000); a = a_amount(net, cfg["scale"])
        miss_b.append(a)
        A.append(dict(order_id=f"ORD-M{k:04d}", order_date=(OCT1 + timedelta(days=rnd.randrange(0, 30))).isoformat(),
                      customer_email=f"m{k}@example.com", region="AMER", order_total=f"{a / 100:.2f}", order_status="completed"))
    miss_a = []
    for k in range(cfg["missing_a"]):
        v = rnd.randint(5000, 150000); miss_a.append(v)
        B.append(dict(txn_ref=f"JNL-{k:04d}", posting_date=(OCT1 + timedelta(days=rnd.randrange(0, 30))).isoformat(),
                      amount=f"{v / 100:.2f}", ledger_status="posted"))
    for k in range(100):                                       # realistic out-of-period rows in both
        net = rnd.randint(2000, 90000); d = (date(2026, 11, 3) + timedelta(days=rnd.randrange(0, 17))).isoformat()
        A.append(dict(order_id=f"ORD-N{k:04d}", order_date=d, customer_email=f"n{k}@example.com", region="APAC",
                      order_total=f"{a_amount(net, cfg['scale']) / 100:.2f}", order_status="completed"))
        B.append(dict(txn_ref=f"ORD-N{k:04d}", posting_date=d, amount=f"{net / 100:.2f}", ledger_status="posted"))
    rnd.shuffle(A); rnd.shuffle(B)

    # ---- independent ground truth (A -> B bridge contributions, cents) ----
    fam = {
        "duplicates": -sum(amt[i][0] for i in dup_i),
        "status_scope": -sum(canc),
        "cutoff": -sum(amt[i][0] for i in cut_i) + sum(amt[i][0] for i in prior_i),
        "missing_records": -sum(miss_b) + sum(miss_a),
    }
    matched = [i for i in range(n) if i not in cut_i]
    fam["value_basis"] = sum(amt[i][1] - amt[i][0] - amt[i][2] for i in matched)      # tax/rounding part only
    unexplained = sum(amt[i][2] for i in matched)
    def rep(rows, col_d, col_a):
        return sum(round(float(r[col_a]) * 100) for r in rows if OCT1.isoformat() <= r[col_d] <= OCT31.isoformat())
    gap = rep(B, "posting_date", "amount") - rep(A, "order_date", "order_total")
    assert sum(fam.values()) + unexplained == gap, (name, fam, unexplained, gap)
    d = pathlib.Path(__file__).parent / "scenarios" / name
    d.mkdir(parents=True, exist_ok=True)
    for fn, rows in (("a.csv", A), ("b.csv", B)):
        with (d / fn).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (d / "truth.json").write_text(json.dumps({
        "label": "October 2026 revenue", "period": [OCT1.isoformat(), OCT31.isoformat()],
        "families": {k: v for k, v in fam.items() if v != 0}, "unexplained": unexplained, "gap": gap,
        "planted": cfg}, indent=2))
    return gap


if __name__ == "__main__":
    for i, (nm, cfg) in enumerate(SCENARIOS.items()):
        print(nm, "gap cents:", build(nm, cfg, seed=100 + i))
