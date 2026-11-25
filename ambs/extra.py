"""Extended-analysis tables computed from saved outputs and verdicts (no API calls), plus a throughput benchmark.

  python -m ambs.extra significance     # AMBS-IC vs Base: paired bootstrap CIs, empty-response rate, length
  python -m ambs.extra interference     # LLaMA-2-7B: each pathway on all three benchmarks, lambda = 0 / 0.1 / 1
  python -m ambs.extra depth            # constant offsets at l = 1, L/2, L under joint decoding (mid_joint runs)
  python -m ambs.extra instruct         # instruction-tuned backbones (Table 1 protocol)
  python -m ambs.extra baselines        # matched baselines (ambs/baselines.py)
  python -m ambs.extra targets          # supervision-target quality
  python -m ambs.extra bench --backbone llama2-7b   # GPU: decoding throughput, base vs AMBS vs AMBS-IC vs adapters
  python -m ambs.extra efficiency       # parameters, training time, throughput
  python -m ambs.extra new              # all builders below
  python -m ambs.extra interf0          # pathway x benchmark at lambda = 0, all four backbones
  python -m ambs.extra mech             # single rank-24 affine offset vs AMBS / AMBS-IC (3 seeds, sub200)
  python -m ambs.extra seeds            # mean ± sd over training seeds 42-44, every method (sub200)
  python -m ambs.extra ctrl             # controllability: decoding-time pathway weights of AMBS-IC (sub200)
  python -m ambs.extra rep              # representation-intervention baselines beside the matched baselines (sub200)
  python -m ambs.extra capability       # MMLU, GSM8K, XSTest over-refusal
  python -m ambs.extra ood              # MT-Bench, HaluEval, AdvBench
  python -m ambs.extra ontopic          # on-topic safety, refusal, off-topic on BeaverTails
  python -m ambs.extra jointresp        # same-response joint audit (48 prompts)
  python -m ambs.extra deploy           # explicit vs folded AMBS-IC, speed and memory
Outputs: $AMBS_ROOT/results/ext_<name>.md
"""
import argparse
import json
import os
import random
import statistics
import sys

from .common import BACKBONES, CHAT_BACKBONES, RUNS, read_jsonl
from .experiments import NUM_LAYERS, R, gen_tag
from .tables import BB_LABEL, metrics, safe

OUT = os.path.join(os.path.dirname(RUNS), "results")
C, IC = R(), R(variant="ic")
f1 = lambda v: "—" if v is None else f"{v:.1f}"


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def write(name, lines):
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/ext_{name}.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _items(bb, tag, judge_tag=""):
    d = f"{RUNS}/{bb}/gen/{tag}"
    j, s = jl(f"{d}/judge_items{judge_tag}.json"), jl(f"{d}/safety_items.json")
    if not (j and s):
        return None
    return ({x["id"]: x["result"] == "win" for x in j["alpacaeval"]},
            {x["id"]: (bool(x["truthful"]), bool(x["informative"])) for x in j["truthfulqa"]},
            {x["id"]: bool(x["flagged"]) for x in s})


def significance(B=10000):
    L = ["| Backbone | ΔWR [95% CI] | ΔSafe [95% CI] | ΔTI_prod [95% CI] | ΔS_HHH [95% CI] | Empty % (Base / IC) | Words (Base / IC) |",
         "|---|---|---|---|---|---|---|"]
    rng = random.Random(0)
    for bb in BACKBONES:
        a, b = _items(bb, "base"), _items(bb, gen_tag(IC, "joint"))
        if not (a and b):
            L.append(f"| {BB_LABEL[bb]} | — | — | — | — | — | — |"); continue
        ids = [sorted(set(a[k]) & set(b[k])) for k in range(3)]

        def stat(sel):
            w = [100 * statistics.mean([b[0][i] - a[0][i] for i in sel[0]])][0]
            tb = (statistics.mean([b[1][i][0] for i in sel[1]]), statistics.mean([b[1][i][1] for i in sel[1]]))
            ta = (statistics.mean([a[1][i][0] for i in sel[1]]), statistics.mean([a[1][i][1] for i in sel[1]]))
            t = 100 * (tb[0] * tb[1] - ta[0] * ta[1])
            s = 100 * statistics.mean([a[2][i] - b[2][i] for i in sel[2]])  # change in Safe = -change in unsafe rate
            return w, s, t, (w + t + s) / 3
        point = stat(ids)
        boots = [stat([[rng.choice(x) for _ in x] for x in ids]) for _ in range(B)]
        ci = [sorted(v[k] for v in boots) for k in range(4)]
        cs = [f"{point[k]:+.1f} [{ci[k][int(0.025 * B)]:+.1f}, {ci[k][int(0.975 * B)]:+.1f}]" for k in range(4)]
        outs = {}
        for name, tag in (("base", "base"), ("ic", gen_tag(IC, "joint"))):
            rows = [r for bn in ("alpacaeval", "truthfulqa", "beavertails")
                    for r in read_jsonl(f"{RUNS}/{bb}/gen/{tag}/{bn}.jsonl")]
            outs[name] = (100 * sum(not r["output"].strip() for r in rows) / len(rows),
                          statistics.mean(len(r["output"].split()) for r in rows))
        L.append(f"| {BB_LABEL[bb]} | {' | '.join(cs)} | {outs['base'][0]:.1f} / {outs['ic'][0]:.1f} | "
                 f"{outs['base'][1]:.0f} / {outs['ic'][1]:.0f} |")
    L += ["", f"AMBS-IC (joint) minus Base; {B} paired bootstrap resamples of prompts within each benchmark."]
    write("significance", L)


def interference():
    L = ["| λ | Pathway | WR | Safe | TI_prod |", "|---|---|---|---|---|"]
    for lam in (0.0, 0.1, 1.0):
        r = R(lam=lam)
        for p, lab in (("path-help", "Helpfulness"), ("path-harm", "Harmlessness"), ("path-honest", "Honesty")):
            m = metrics("llama2-7b", gen_tag(r, p))
            L.append(f"| {lam} | {lab} | {f1(m['WR'])} | {f1(safe(m))} | {f1(m['TI'])} |")
    L += ["", "LLaMA-2-7B; every pathway evaluated on all three benchmarks (diagonal = its own objective)."]
    write("interference", L)


def _grid(name, bbs, rows, label, judge_tag="", note=""):
    L = [f"| {label} | " + " | ".join(f"{BB_LABEL.get(b, b)} WR | Safe | TI_prod | S_HHH" for b in bbs) + " |",
         "|" + "---|" * (1 + 4 * len(bbs))]
    for lab, tagf in rows:
        cells = []
        for b in bbs:
            m = metrics(b, tagf(b), judge_tag) if tagf(b) else {"WR": None, "SS": None, "TI": None, "Avg": None}
            cells += [f1(m["WR"]), f1(safe(m)), f1(m["TI"]), f1(m["Avg"])]
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    write(name, L + (["", note] if note else []))


def depth():
    _grid("depth", list(BACKBONES), [
        ("l=1", lambda b: gen_tag(R(layer=1), "joint")),
        ("l=L/2", lambda b: gen_tag(R(layer=NUM_LAYERS[b] // 2), "joint")),
        ("l=L (default)", lambda b: gen_tag(C, "joint"))], "Layer")


def instruct():
    _grid("instruct", list(CHAT_BACKBONES), [
        ("Base (no AMBS)", lambda b: "base"), ("AMBS (A_joint)", lambda b: gen_tag(C, "joint")),
        ("AMBS-IC (A_joint)", lambda b: gen_tag(IC, "joint"))], "Method")


def baselines():
    _grid("baselines", list(BACKBONES), [("Base", lambda b: "base")] +
          [(m, (lambda mm: lambda b: f"{mm}-s42__joint")(m)) for m in ("caa", "lora_sft", "dpo")] +
          [("AMBS (A_joint)", lambda b: gen_tag(C, "joint")), ("AMBS-IC (A_joint)", lambda b: gen_tag(IC, "joint"))],
          "Method")


def targets():
    fam = lambda b: "llama2" if b.startswith("llama2") else "mistral"
    bbs = ["llama2-7b", "llama2-7b-chat", "mistral-7b", "mistral-7b-instruct"]
    L = ["| Targets | " + " | ".join(BB_LABEL.get(b, b) for b in bbs) + " |", "|---|---|---|---|---|"]
    for lab, tf in (("No offset", lambda b: "base"), ("Original", lambda b: gen_tag(C, "joint")),
                    ("Stronger", lambda b: gen_tag(R(data_tag="strong"), "joint")),
                    ("Self-distilled", lambda b: gen_tag(R(data_tag=f"self_{fam(b)}"), "joint"))):
        L.append(f"| {lab} | " + " | ".join(f1(metrics(b, tf(b))["Avg"]) for b in bbs) + " |")
    L += ["", "Joint S_HHH = (WR + TI_prod + Safe) / 3, Safe = 100 - SS."]
    write("targets", L)


def bench(bb, n=64):
    """Decoding throughput (generated tokens / s) on n AlpacaEval prompts, batch 16, 256 new tokens."""
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .common import ALL_BACKBONES, DATA, format_prompt
    from .generate import load_offsets
    from .model import install_hook
    tok = AutoTokenizer.from_pretrained(ALL_BACKBONES[bb], padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    prompts = [format_prompt(r["instruction"]) for r in read_jsonl(f"{DATA}/eval_alpacaeval.jsonl")[:n]]
    dt = getattr(torch, open(f"{RUNS}/{bb}/DTYPE").read().strip()) if os.path.exists(f"{RUNS}/{bb}/DTYPE") else torch.float16
    res = {}

    def run(model):
        torch.manual_seed(0); toks = 0; torch.cuda.synchronize(); t0 = time.time()
        for s in range(0, n, 16):
            enc = tok(prompts[s:s + 16], return_tensors="pt", padding=True).to("cuda")
            out = model.generate(**enc, do_sample=False, max_new_tokens=256, min_new_tokens=256, pad_token_id=tok.pad_token_id)
            toks += (out.shape[1] - enc.input_ids.shape[1]) * out.shape[0]
        torch.cuda.synchronize()
        return toks / (time.time() - t0)
    base = AutoModelForCausalLM.from_pretrained(ALL_BACKBONES[bb], torch_dtype=dt, device_map="cuda").eval()
    with torch.no_grad():
        run(base); res["Base"] = run(base)
        for lab, r in (("AMBS", C), ("AMBS-IC", IC)):
            off, layer = load_offsets(f"{RUNS}/{bb}/{gen_tag(r, 'joint').split('__')[0]}", base.config.hidden_size,
                                      base.get_output_embeddings().weight.shape[0])
            h = install_hook(base, off, [1 / off.n] * off.n, layer=layer); res[lab] = run(base); h.remove()
    from peft import PeftModel
    for m in ("lora_sft", "dpo"):
        d = f"{RUNS}/{bb}/{m}-s42"
        if os.path.exists(f"{d}/adapter_config.json"):
            pm = PeftModel.from_pretrained(base, d).merge_and_unload().eval()
            with torch.no_grad():
                res[m] = run(pm)
            base = AutoModelForCausalLM.from_pretrained(ALL_BACKBONES[bb], torch_dtype=dt, device_map="cuda").eval()
    json.dump({"backbone": bb, "tokens_per_sec": res, "gpu": torch.cuda.get_device_name()},
              open(f"{OUT}/ext_bench_{bb}.json", "w"), indent=1)
    print(res)


def efficiency(bb="llama2-7b"):
    d, N, r = 4096, 3, 8
    b = jl(f"{OUT}/ext_bench_{bb}.json") or {"tokens_per_sec": {}}
    tps = b["tokens_per_sec"]
    rel = lambda k: f"{tps[k] / tps['Base']:.2f}" if k in tps and "Base" in tps else "—"

    def sec(name):
        lg = f"{RUNS}/{bb}/{name}/train_log.jsonl"
        rows = read_jsonl(lg) if os.path.exists(lg) else []
        return f"{rows[-1]['sec'] / 3600:.2f}" if rows else "—"
    done = {m: jl(f"{RUNS}/{bb}/{m}-s42/done.json") or {} for m in ("lora_sft", "dpo")}
    L = ["| Method | Trainable params | Training (GPU-h) | Throughput (rel. to Base) |", "|---|---|---|---|",
         f"| Base | 0 | — | 1.00 |",
         f"| AMBS | {N * (2 * d * r + d):,} | {sec(gen_tag(C, 'joint').split('__')[0])} | {rel('AMBS')} |",
         f"| AMBS-IC | {N * (2 * d * r + d):,} | {sec(gen_tag(IC, 'joint').split('__')[0])} | {rel('AMBS-IC')} |"]
    for m, lab in (("lora_sft", "LoRA-SFT"), ("dpo", "DPO")):
        dd = done[m]
        L.append(f"| {lab} | {dd.get('trainable_params', '—'):,} | "
                 f"{(dd['train_sec'] / 3600):.2f} | {rel(m)} |" if dd else f"| {lab} | — | — | {rel(m)} |")
    L += ["", f"{BB_LABEL[bb]}; AMBS training excludes the one-off hidden-state cache; throughput on {b.get('gpu', '—')}."]
    write("efficiency", L)


# ---------------------------------------------------------------- additional analyses
SUB = "_sub200"  # WR and TI on the fixed 200-prompt subsets (seed 42); Safe on the full BeaverTails set
SUB_NOTE = "WR and TI_prod on the fixed 200-prompt subsets (seed 42) of AlpacaEval and TruthfulQA; Safe on the full set."
EV = f"{OUT}/evals"
BBS = list(BACKBONES)
SEEDS = (42, 43, 44)
SYS_LABEL = {"base": "Base", "caa": "CAA", "lora_sft": "LoRA-SFT", "dpo": "DPO", "ambs": "AMBS (A_joint)",
             "ambs_ic": "AMBS-IC (A_joint)", "affine": "Single affine (rank 24)"}
AFFINE = lambda s: R(variant="ic", rank=24, scheme="pooled", seed=s)
f2 = lambda v: "—" if v is None else f"{v:.2f}"


def _seed_grid(name, rows, note, extra=()):
    """rows: (label, tag_fn(bb, seed)); cells are mean ± sd over SEEDS, '—' unless every seed is scored."""
    L = ["| Method | " + " | ".join(f"{BB_LABEL[b]} WR | Safe | TI_prod | S_HHH" for b in BBS) + " |",
         "|" + "---|" * (1 + 4 * len(BBS))]
    for lab, tf in rows:
        cells = []
        for b in BBS:
            tags = [tf(b, s) for s in SEEDS]
            ms = [metrics(b, t, SUB) for t in dict.fromkeys(tags)]  # one model (e.g. Base): a single score, no sd
            for k in ("WR", "Safe", "TI", "Avg"):
                v = [safe(m) if k == "Safe" else m[k] for m in ms]
                cells.append("—" if None in v else f1(v[0]) if len(v) == 1 else
                             f"{statistics.mean(v):.1f} ± {statistics.stdev(v):.1f}")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    write(name, L + list(extra) + ["", note])


def _ev_grid(name, kind, systems, cols, note):
    """Wide table from results/evals/<kind>_<bb>_<system>.json; cols: (header, key, formatter)."""
    L = ["| System | " + " | ".join(" | ".join(f"{BB_LABEL[b]} {h}" if i == 0 else h for i, (h, _, _) in enumerate(cols))
                                   for b in BBS) + " |", "|" + "---|" * (1 + len(cols) * len(BBS))]
    for sy in systems:
        cells = []
        for b in BBS:
            d = jl(f"{EV}/{kind}_{b}_{sy}.json") or {}
            cells += [fm(d.get(k)) for _, k, fm in cols]
        L.append(f"| {SYS_LABEL[sy]} | " + " | ".join(cells) + " |")
    write(name, L + ["", note])


def interf0():
    _grid("interf0", BBS, [(f"{lab} pathway only", (lambda pp: lambda b: gen_tag(R(lam=0.0), pp))(p))
                           for p, lab in (("path-help", "Helpfulness"), ("path-harm", "Harmlessness"),
                                          ("path-honest", "Honesty"))], "λ = 0",
          note="Constant AMBS trained without the consistency term (λ = 0); each pathway alone decoded on all three "
               "benchmarks (full sets). LLaMA-2-7B rows are the λ = 0 runs of the interference experiment.")


def _seed_mean_items(bb, tags):
    """Per-prompt labels averaged over seeds (sub200 judge items); None unless every seed is scored."""
    its = [_items(bb, t, SUB) for t in tags]
    if not all(its):
        return None
    ids = [sorted(set.intersection(*(set(x[k]) for x in its))) for k in range(3)]
    m = lambda k, i, f: statistics.mean(f(x[k][i]) for x in its)
    return ({i: m(0, i, float) for i in ids[0]},
            {i: (m(1, i, lambda v: float(v[0])), m(1, i, lambda v: float(v[1]))) for i in ids[1]},
            {i: m(2, i, float) for i in ids[2]})


def paired_diff(a, b, B=10000, seed=0):
    """b minus a: WR, Safe, TI_prod, S_HHH with 95% paired bootstrap CIs over prompts within each benchmark."""
    ids = [sorted(set(a[k]) & set(b[k])) for k in range(3)]

    def stat(sel):
        w = 100 * statistics.mean(b[0][i] - a[0][i] for i in sel[0])
        tp = lambda x: statistics.mean(x[1][i][0] for i in sel[1]) * statistics.mean(x[1][i][1] for i in sel[1])
        t = 100 * (tp(b) - tp(a))
        s = 100 * statistics.mean(a[2][i] - b[2][i] for i in sel[2])
        return w, s, t, (w + s + t) / 3
    rng = random.Random(seed)
    point = stat(ids)
    draws = [stat([[rng.choice(x) for _ in x] for x in ids]) for _ in range(B)]
    cols = [sorted(v[k] for v in draws) for k in range(4)]
    # 95% interval, plus the Bonferroni-adjusted 98.75% interval for four backbones (alpha = 0.05 / 4)
    return [(point[k], cols[k][int(0.025 * B)], cols[k][int(0.975 * B)],
             cols[k][int(0.00625 * B)], cols[k][int(0.99375 * B)]) for k in range(4)]


def mech():
    out = {}
    for b in BBS:
        a = _seed_mean_items(b, [gen_tag(AFFINE(s), "joint") for s in SEEDS])
        f = _seed_mean_items(b, [gen_tag(R(variant="ic", seed=s), "joint") for s in SEEDS])
        out[b] = paired_diff(a, f) if a and f else None
    json.dump(out, open(f"{OUT}/ext_mech_ci.json", "w"), indent=1)
    ci_row = "| AMBS-IC minus single affine [95% CI] | " + " | ".join(
        " | ".join(f"{p:+.1f} [{lo:+.1f}, {hi:+.1f}]" for p, lo, hi, *_ in out[b]) if out[b] else " | ".join(["—"] * 4)
        for b in BBS) + " |"
    _seed_grid("mech", [("Base", lambda b, s: "base"),
                        ("Single affine (rank 24, mixed targets)", lambda b, s: gen_tag(AFFINE(s), "joint")),
                        ("AMBS (constant, A_joint)", lambda b, s: gen_tag(R(seed=s), "joint")),
                        ("AMBS-IC (A_joint)", lambda b, s: gen_tag(R(variant="ic", seed=s), "joint"))],
               "Mean ± sd over training seeds 42, 43, 44 (Base: one model, single score). "
               "The single affine offset matches AMBS-IC's total rank (N·r = 24) but has one pooled map. " + SUB_NOTE +
               " CI: seed-averaged labels per prompt, 10,000 paired bootstrap resamples within each benchmark.", [ci_row])


def seeds():
    _seed_grid("seeds", [(SYS_LABEL[m], (lambda mm: lambda b, s: f"{mm}-s{s}__joint")(m)) for m in ("caa", "lora_sft", "dpo")] +
               [("AMBS (A_joint)", lambda b, s: gen_tag(R(seed=s), "joint")),
                ("AMBS-IC (A_joint)", lambda b, s: gen_tag(R(variant="ic", seed=s), "joint"))],
               "Mean ± sd over training seeds 42, 43, 44 for every trained method. " + SUB_NOTE)


def ctrl():
    run = gen_tag(IC, "joint").split("__")[0]
    rows = [("w = (1/3, 1/3, 1/3) (Table 1)", lambda b: f"{run}__joint")]
    rows += [(f"w = ({w.replace(',', ', ')})", (lambda ww: lambda b: f"{run}__w{ww.replace(',', '-')}")(w))
             for w in ("0.6,0.2,0.2", "0.2,0.6,0.2", "0.2,0.2,0.6")]
    rows += [(f"w = {v} ({lab} only)", (lambda mm: lambda b: f"{run}__{mm}")(m)) for m, v, lab in
             (("path-help", "(1, 0, 0)", "help"), ("path-harm", "(0, 1, 0)", "harm"), ("path-honest", "(0, 0, 1)", "honest"))]
    _grid("ctrl", BBS, rows, "Pooling weights (help, harm, honest)", SUB,
          "One trained AMBS-IC model per backbone, re-weighted at decoding time (no retraining). " + SUB_NOTE)
    ws = [(1 / 3,) * 3, (.6, .2, .2), (.2, .6, .2), (.2, .2, .6), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    L = ["| Spearman rho (weight vs own metric) | " + " | ".join(f"{BB_LABEL[b]} WR | Safe | TI_prod" for b in BBS) + " |",
         "|" + "---|" * (1 + 3 * len(BBS)), "| rho |"]
    for b in BBS:
        ms = [metrics(b, tf(b), SUB) for _, tf in rows]
        for k, key in enumerate(("WR", "Safe", "TI")):
            v = [safe(m) if key == "Safe" else m[key] for m in ms]
            L[-1] += " — |" if None in v else f" {spearman([w[k] for w in ws], v):.2f} |"
    write("ctrl_spearman", L + ["", "Seven weightings of the ctrl table; ties get average ranks."])


def _ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    c = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return c / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


def rep():
    rows = [("Base", lambda b: "base")]
    rows += [(lab, (lambda mm: lambda b: f"{mm}-s42__joint")(m)) for m, lab in
             (("caa", "CAA"), ("caa_tuned", "CAA (tuned)"), ("iti", "ITI"), ("loreft", "LoReFT"),
              ("matsteer", "MAT-Steer"), ("lora_sft", "LoRA-SFT"), ("dpo", "DPO"))]
    rows += [("AMBS (A_joint)", lambda b: gen_tag(C, "joint")), ("AMBS-IC (A_joint)", lambda b: gen_tag(IC, "joint"))]
    _grid("rep", BBS, rows, "Method", SUB, "Seed 42 for every trained method. " + SUB_NOTE)


def capability():
    _ev_grid("capability", "capability", ("base", "lora_sft", "dpo", "ambs", "ambs_ic"),
             [("MMLU", "mmlu", f1), ("GSM8K", "gsm8k", f1), ("XSTest refusal ↓", "xstest_refusal", f1)],
             "MMLU 2,000 questions (5-shot), GSM8K 500 problems (8-shot, greedy, exact match), XSTest 250 safe prompts "
             "(refusal rate by string match; lower is better). Accuracy in %.")


def ood():
    _ev_grid("ood", "ood", ("base", "caa", "lora_sft", "dpo", "ambs_ic"),
             [("MT-Bench", "mtbench", f2), ("HaluEval", "halueval_acc", f1), ("AdvBench Safe", "advbench_safe", f1)],
             "MT-Bench: GPT-4o 1-10 rating, 80 first-turn questions. HaluEval QA: GPT-4o correctness, 500 items (%). "
             "AdvBench: 100 − Beaver-Dam-7B unsafe rate, 520 behaviours.")


def ontopic():
    _ev_grid("ontopic", "ontopic", ("base", "dpo", "ambs", "ambs_ic"),
             [("Safe & on-topic", "safe_on_topic", f1), ("Unsafe ↓", "unsafe", f1), ("Refusal", "refusal", f1),
              ("Off-topic ↓", "off_topic", f1)],
             "500 BeaverTails test prompts (seed 42). Unsafe: Beaver-Dam-7B; on-topic and refusal: GPT-4o. In %.")


def jointresp():
    keys = [("Joint benign", "joint_benign"), ("Joint unsafe", "joint_unsafe"), ("Joint macro", "joint_macro"),
            ("Benign refusal ↓", "benign_refusal"), ("Factual error ↓", "factual_error")]
    L = ["| Backbone | System | " + " | ".join(h for h, _ in keys) + " |", "|---|---|" + "---|" * len(keys)]
    for b in BBS:
        d = jl(f"{EV}/jointresp_{b}.json") or {}
        for sy in ("base", "affine", "ambs_ic"):
            L.append(f"| {BB_LABEL[b]} | {SYS_LABEL[sy]} | " + " | ".join(f1(d.get(sy, {}).get(k)) for _, k in keys) + " |")
        ci = d.get("diff_ci", {})
        L.append(f"| {BB_LABEL[b]} | Δ AMBS-IC − affine [95% CI] | " + " | ".join(
            f"{ci[k][0]:+.1f} [{ci[k][1]:+.1f}, {ci[k][2]:+.1f}]" if k in ci else "—" for _, k in keys) + " |")
    write("jointresp", L + ["", "48 prompts (24 TruthfulQA benign, 24 BeaverTails unsafe, seed 42). A response is jointly "
                            "acceptable if it is on-topic (GPT-4o), safe (Beaver-Dam-7B), factually correct (GPT-4o) and "
                            "non-empty. CI: 10,000 paired bootstrap resamples within strata. In %."])


def deploy():
    keys = [("Tokens/s", "tok_s", lambda v: "—" if v is None else f"{v:,.0f}"), ("p95 ms/token", "p95_ms_token", f2),
            ("Peak VRAM (GB)", "peak_vram_gb", f2), ("Max abs Δlogit vs explicit", "max_logit_diff", lambda v: "—" if v is None else f"{v:.4f}"),
            ("Top-1 match (%)", "token_match", f1)]
    L = ["| Backbone | Variant | " + " | ".join(h for h, _, _ in keys) + " |", "|---|---|" + "---|" * len(keys)]
    gpus = set()
    for b in BBS:
        d = jl(f"{EV}/deploy_{b}.json") or {}
        gpus.add(d.get("gpu"))
        for v, lab in (("base", "Base"), ("explicit", "AMBS-IC, explicit branches"), ("folded", "AMBS-IC, folded affine")):
            L.append(f"| {BB_LABEL[b]} | {lab} | " + " | ".join(fm(d.get(v, {}).get(k)) for _, k, fm in keys) + " |")
    write("deploy", L + ["", f"BF16, batch 16, 256 greedy tokens; p95 latency over 64 cached decoding steps; logit check on "
                            f"64 AlpacaEval prompts. GPU: {', '.join(sorted(g for g in gpus if g)) or '—'}."])


NEW = ("interf0", "mech", "seeds", "ctrl", "rep", "capability", "ood", "ontopic", "jointresp", "deploy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["significance", "interference", "depth", "instruct", "baselines", "targets",
                                    "bench", "efficiency", "new", *NEW])
    ap.add_argument("--backbone", default="llama2-7b")
    a = ap.parse_args()
    if a.cmd == "bench":
        bench(a.backbone)
    elif a.cmd == "efficiency":
        efficiency(a.backbone)
    elif a.cmd == "new":
        for n in NEW:
            globals()[n]()
    else:
        globals()[a.cmd]()


if __name__ == "__main__":
    main()
