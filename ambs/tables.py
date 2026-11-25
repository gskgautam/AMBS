"""Build the paper's tables from score files.  python -m ambs.tables [--exp table1 ...]   ->  results/<exp>.md (+ .csv)

Missing scores are shown as "—", so tables can be rebuilt at any time while experiments are still running.
"""
import argparse
import csv
import itertools
import json
import os
import random
import statistics

from .common import BACKBONES, CHAT_BACKBONES, RUNS
from .experiments import R, gen_tag, plan

RES = os.path.join(os.path.dirname(RUNS), "results")
BBS = list(BACKBONES)  # Tables 1-8: the four base backbones (instruction-tuned models: python -m ambs.extra instruct)
BB_LABEL = {"llama2-7b": "LLaMA-2-7B", "mistral-7b": "Mistral-7B", "gemma-7b": "Gemma-7B", "deepseek-7b": "DeepSeek-7B",
            "llama2-7b-chat": "LLaMA-2-7B-chat"}


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def metrics(bb, tag, judge_tag=""):
    """WR, TI_prod, SS, Safe = 100 - SS and the HHH score (key "Avg") of one generation folder; None if not scored."""
    d = f"{RUNS}/{bb}/gen/{tag}"
    j, s = jload(f"{d}/judge_scores{judge_tag}.json"), jload(f"{d}/safety_scores.json")
    m = {"WR": j["alpacaeval"]["WR"] if j and "alpacaeval" in j else None,
         "TI": j["truthfulqa"]["TI_prod"] if j and "truthfulqa" in j else None,
         "SS": s["SS"] if s else None}
    m["Avg"] = avg(m)
    return m


def avg(m):
    """HHH score S_HHH = (WR + TI_prod + Safe) / 3 with Safe = 100 - SS; every term is higher-is-better on 0-100."""
    return None if None in (m["WR"], m["TI"], m["SS"]) else (m["WR"] + m["TI"] + (100 - m["SS"])) / 3


def safe(m):
    return None if m.get("SS") is None else 100 - m["SS"]


def f1(v):
    return "—" if v is None else f"{v:.1f}"


def msd(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return "—"
    return f"{statistics.mean(vals):.1f}±{statistics.stdev(vals):.1f}"


def write(name, lines, rows=None, note=""):
    os.makedirs(RES, exist_ok=True)
    open(f"{RES}/{name}.md", "w").write("\n".join(lines) + ("\n\n" + note if note else "") + "\n")
    if rows:
        with open(f"{RES}/{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print("\n".join(lines)); print()


def table1():
    C, IC = R(), R(variant="ic")
    head = "| Method | " + " | ".join(f"{BB_LABEL[b]} WR | Safe | TI_prod | S_HHH" for b in BBS) + " |"
    lines = [head, "|" + "---|" * (1 + 4 * len(BBS))]
    rows = []
    def add(label, per_bb):
        cells = []
        for b in BBS:
            m = per_bb(b)
            cells += [f1(m["WR"]), f1(safe(m)), f1(m["TI"]), f1(m["Avg"])]
            rows.append({"method": label, "backbone": b, **m})
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    add("Base", lambda b: metrics(b, "base"))
    for pw, lab in (("help", "Helpfulness"), ("harm", "Harmlessness"), ("honest", "Honesty")):
        add(f"AMBS ({lab})", lambda b, pw=pw: metrics(b, gen_tag(C, f"path-{pw}")))
    def asep(b):
        m = {"WR": metrics(b, gen_tag(C, "path-help"))["WR"], "SS": metrics(b, gen_tag(C, "path-harm"))["SS"],
             "TI": metrics(b, gen_tag(C, "path-honest"))["TI"]}
        m["Avg"] = avg(m); return m
    add("AMBS (A_sep)", asep)
    add("AMBS (A_joint)", lambda b: metrics(b, gen_tag(C, "joint")))
    add("AMBS-IC (A_joint)", lambda b: metrics(b, gen_tag(IC, "joint")))
    write("table1", lines, rows, "Targeted-decoding and multi-objective baselines are source-reported in the paper "
          "(not rerun). A_sep = WR of the helpfulness pathway, SS of the harmlessness pathway, TI of the honesty pathway.")


def table2():
    C = R()
    lines = ["| Backbone | Joint WR | Safe | TI_prod | S_HHH | A_sep (seed 42) | Seed 43 | Seed 44 | Mean±SD | AMBS-IC A_joint seeds 42/43/44 | Mean±SD |",
             "|" + "---|" * 11]
    rows = []
    for b in BBS:
        j = metrics(b, gen_tag(C, "joint"))
        s42 = {"WR": metrics(b, gen_tag(C, "path-help"))["WR"], "SS": metrics(b, gen_tag(C, "path-harm"))["SS"],
               "TI": metrics(b, gen_tag(C, "path-honest"))["TI"]}
        seps = [avg(s42)] + [metrics(b, gen_tag(R(seed=s), "sep"))["Avg"] for s in (43, 44)]
        ics = [metrics(b, gen_tag(R(variant="ic", seed=s), "joint"))["Avg"] for s in (42, 43, 44)]
        lines.append(f"| {BB_LABEL[b]} | {f1(j['WR'])} | {f1(safe(j))} | {f1(j['TI'])} | {f1(j['Avg'])} | "
                     + " | ".join(f1(v) for v in seps) + f" | {msd(seps)} | " + " / ".join(f1(v) for v in ics) + f" | {msd(ics)} |")
        rows.append({"backbone": b, "joint_avg": j["Avg"], "asep_seeds": seps, "ic_joint_seeds": ics})
    write("table2", lines, rows, "Sample SD (K-1), K = 3 seeds.")


def simplex(mA, mB, step=0.05):
    """Delta_{A,B}(alpha) = A_alpha(A) - A_alpha(B), A_alpha = a1 WR + a2 TI - a3 SS, over the 231-point grid."""
    n = round(1 / step)
    ds = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            a1, a2, a3 = i / n, j / n, (n - i - j) / n
            sc = lambda m: a1 * m["WR"] + a2 * m["TI"] - a3 * m["SS"]
            ds.append(sc(mA) - sc(mB))
    return ds


def table3():
    C, D = R(), R(variant="direct_vector", wd=0.0)
    lines = ["| Backbone | AMBS Ref. S_HHH | Pert. S_HHH | Δ | Direct-Vector Ref. S_HHH | Pert. S_HHH | Δ | Min Δ_A,D | Max Δ_A,D | A≥D (%) | A≥C (%) |",
             "|" + "---|" * 11]
    rows = []
    for b in BBS:
        a, ap_ = metrics(b, gen_tag(C, "joint")), metrics(b, gen_tag(R(data_tag="pert"), "joint"))
        d, dp = metrics(b, gen_tag(D, "joint")), metrics(b, gen_tag(R(variant="direct_vector", wd=0.0, data_tag="pert"), "joint"))
        c = metrics(b, gen_tag(R(variant="direct_vector", wd=0.0, scheme="pooled"), "joint"))
        dl = lambda x, y: None if None in (x["Avg"], y["Avg"]) else y["Avg"] - x["Avg"]
        mn = mx = ge_d = ge_c = None
        if a["Avg"] is not None and d["Avg"] is not None:
            ds = simplex(a, d); mn, mx = min(ds), max(ds); ge_d = 100 * sum(x >= -1e-9 for x in ds) / len(ds)
        if a["Avg"] is not None and c["Avg"] is not None:
            dc = simplex(a, c); ge_c = 100 * sum(x >= -1e-9 for x in dc) / len(dc)
        lines.append(f"| {BB_LABEL[b]} | {f1(a['Avg'])} | {f1(ap_['Avg'])} | {f1(dl(a, ap_))} | {f1(d['Avg'])} | "
                     f"{f1(dp['Avg'])} | {f1(dl(d, dp))} | {f1(mn)} | {f1(mx)} | {f1(ge_d)} | {f1(ge_c)} |")
        rows.append({"backbone": b, "A": a, "A_pert": ap_, "D": d, "D_pert": dp, "C": c,
                     "min_dAD": mn, "max_dAD": mx, "A_ge_D_pct": ge_d, "A_ge_C_pct": ge_c})
    write("table3", lines, rows, "Simplex: 231 evaluation-weight points (step 0.05); A_alpha = a1 WR + a2 TI_prod + a3 Safe (differences equal those of a1 WR + a2 TI_prod - a3 SS).")


def kappa(x, y):
    cats = sorted(set(x) | set(y), key=str)
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, 100 * po


def boot_ci(x, y, B=10000, seed=0):
    rng = random.Random(seed)
    n = len(x)
    pairs = list(zip(x, y))
    vals = sorted(100 * sum(a == b for a, b in (pairs[rng.randrange(n)] for _ in range(n))) / n for _ in range(B))
    return vals[int(0.025 * B)], vals[int(0.975 * B) - 1]


def table4():
    tag = gen_tag(R(), "joint")
    H = {"help": ([], []), "hon": ([], []), "harm": ([], [])}
    sd = {"help": [], "hon": []}
    for b in BBS:
        d = f"{RUNS}/{b}/gen/{tag}"
        g, c = jload(f"{d}/judge_items.json"), jload(f"{d}/judge_items_claude_p0.json")
        if g and c:
            H["help"][0].extend(r["result"] for r in g["alpacaeval"]); H["help"][1].extend(r["result"] for r in c["alpacaeval"])
            for k in ("truthful", "informative"):
                H["hon"][0].extend(r[k] for r in g["truthfulqa"]); H["hon"][1].extend(r[k] for r in c["truthfulqa"])
        bd, lg = jload(f"{d}/safety_items.json"), jload(f"{d}/safety_items_llamaguard3.json")
        if bd and lg:
            H["harm"][0].extend(r["flagged"] for r in bd); H["harm"][1].extend(r["flagged"] for r in lg)
        for judge in ("gpt-4o", "claude"):
            ps = [jload(f"{d}/judge_scores{'' if (judge == 'gpt-4o' and p == 0) else f'_{judge}_p{p}'}.json") for p in (0, 1, 2)]
            if all(ps):
                sd["help"].append(statistics.stdev([p["alpacaeval"]["WR"] for p in ps]))
                sd["hon"].append(statistics.stdev([p["truthfulqa"]["TI_prod"] for p in ps]))
    lines = ["| Objective | Cohen's κ | Agreement (%) | Prompt SD | 95% CI (agreement) |", "|---|---|---|---|---|"]
    rows = []
    for key, lab in (("help", "Helpfulness"), ("hon", "Honesty"), ("harm", "Harmlessness")):
        x, y = H[key]
        if x:
            k, ag = kappa(x, y); lo, hi = boot_ci(x, y)
        else:
            k = ag = lo = hi = None
        s = statistics.mean(sd[key]) if sd.get(key) else None
        lines.append(f"| {lab} | {'—' if k is None else f'{k:.2f}'} | {f1(ag)} | {'n/a' if key == 'harm' else ('—' if s is None else f'{s:.2f}')} | "
                     f"{'—' if lo is None else f'[{lo:.1f}, {hi:.1f}]'} |")
        rows.append({"objective": lab, "kappa": k, "agreement": ag, "prompt_sd": s, "ci_lo": lo, "ci_hi": hi})
    # Overall: all response-level pairs pooled; labels are namespaced by objective so categories never collide
    X = [f"{k}:{v}" for k in ("help", "hon", "harm") for v in H[k][0]]
    Y = [f"{k}:{v}" for k in ("help", "hon", "harm") for v in H[k][1]]
    if X:
        _, ag = kappa(X, Y); lo, hi = boot_ci(X, Y)
        # pooled kappa over namespaced labels would be inflated (chance agreement looks artificially low), so the
        # overall kappa is the per-objective kappa weighted by the number of pairs
        ks = [(kappa(*H[key])[0], len(H[key][0])) for key in ("help", "hon", "harm") if H[key][0]]
        k = sum(kk * n for kk, n in ks) / sum(n for _, n in ks)
        s_all = [v for key in ("help", "hon") for v in sd[key]]
        s = statistics.mean(s_all) if s_all else None
        lines.append(f"| Overall | {k:.2f} | {f1(ag)} | {'—' if s is None else f'{s:.2f}'} | [{lo:.1f}, {hi:.1f}] |")
        rows.append({"objective": "Overall", "kappa": k, "agreement": ag, "prompt_sd": s, "ci_lo": lo, "ci_hi": hi})
    write("table4", lines, rows, "Judges: GPT-4o vs Claude (helpfulness, honesty); Beaver-Dam-7B vs Llama Guard 3 (harmlessness). "
          "Overall: agreement and CI pool all response-level pairs; kappa is the pair-weighted mean of the per-objective kappas; prompt SD averages helpfulness and honesty. "
          "Prompt SD: SD of WR / TI_prod across the 3 prompt formulations (mean over judges and backbones). "
          "CI: 10,000 response-level bootstrap resamples of the agreement rate. Classifiers take no prompt, so no prompt SD for harmlessness.")


def table5():
    cols = [("Base", "base"), ("Direct-Vector", gen_tag(R(variant="direct_vector", wd=0.0), "joint")),
            ("AMBS", gen_tag(R(), "joint")), ("AMBS-IC", gen_tag(R(variant="ic"), "joint"))]
    lines = ["| Backbone | " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * (1 + len(cols))]
    rows, means = [], {c: [] for c, _ in cols}
    for b in BBS:
        vals = []
        for c, t in cols:
            s = jload(f"{RUNS}/{b}/gen/{t}/harmbench_scores.json")
            v = s["ASR"] if s else None
            vals.append(v)
            if v is not None:
                means[c].append(v)
        lines.append(f"| {BB_LABEL[b]} | " + " | ".join(f1(v) for v in vals) + " |")
        rows.append({"backbone": b, **{c: v for (c, _), v in zip(cols, vals)}})
    lines.append("| Mean | " + " | ".join(f"{statistics.mean(v):.2f}" if len(v) == len(BBS) else "—" for v in means.values()) + " |")
    write("table5", lines, rows, "ASR (%), HarmBench DirectRequest, standard + contextual test behaviours, official classifier.")


def sep_table(exp, title_col, judge_tag=""):
    labels = [lab for lab, _, _, _ in plan(exp, BBS[0])]
    lines = [f"| {title_col} | " + " | ".join(BB_LABEL[b] for b in BBS) + " |", "|" + "---|" * (1 + len(BBS))]
    rows = []
    for i, lab in enumerate(labels):
        vals = []
        for b in BBS:
            _, r, _, _ = plan(exp, b)[i]
            vals.append(metrics(b, gen_tag(r, "sep"), judge_tag)["Avg"])
        lines.append(f"| {lab} | " + " | ".join(f1(v) for v in vals) + " |")
        rows.append({"variant": lab, **dict(zip(BBS, vals))})
    return lines, rows


def table6():
    lines, rows = sep_table("table6", "Variant")
    write("table6", lines, rows, "A_sep (objective-specific separate-output evaluation) of constant-offset AMBS variants.")


def figure3():
    lines, rows = sep_table("figure3", "Setting", "_sub200")
    write("figure3", lines, rows, "A_sep for each tested hyperparameter value (one factor varied at a time). WR and TI_prod "
          "are scored on fixed random subsets (seed 42) of 200 AlpacaEval and 200 TruthfulQA prompts; SS on all 3,021 BeaverTails prompts.")


def table8():
    lines = ["| Metric | " + " | ".join(f"{BB_LABEL[b]} Logit-Bias | AMBS" for b in BBS) + " |", "|" + "---|" * (1 + 2 * len(BBS))]
    agg, stab, rows = [], [], []
    for b in BBS:
        lb = [metrics(b, gen_tag(R(variant="logit_bias", lam=0.0, seed=s), "sep"))["Avg"] for s in (42, 43, 44)]
        am = [metrics(b, gen_tag(R(seed=s), "sep"))["Avg"] for s in (42, 43, 44)]
        agg += [f1(lb[0]), f1(am[0])]; stab += [msd(lb), msd(am)]
        rows.append({"backbone": b, "logit_bias": lb, "ambs": am})
    lines += ["| Aggregate HHH (A_sep, seed 42) | " + " | ".join(agg) + " |", "| Mean±SD (3 seeds) | " + " | ".join(stab) + " |"]
    write("table8", lines, rows)


ALL = {"table1": table1, "table2": table2, "table3": table3, "table4": table4, "table5": table5,
       "table6": table6, "figure3": figure3, "table8": table8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all")
    a = ap.parse_args()
    for k, fn in ALL.items():
        if a.exp in ("all", k):
            fn()


if __name__ == "__main__":
    main()
