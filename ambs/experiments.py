"""Every experiment in the paper as a list of runs.  python -m ambs.experiments --exp table6 --backbone llama2-7b --stage gpu

A run = one trained model (or the frozen base) + the decoding modes generated with it.
Stages: gpu   -> hidden-state caches, training, generation, Beaver-Dam / HarmBench / Llama Guard scoring
        judge -> GPT-4o (Batch API) / Claude judging of that experiment's outputs (needs internet)
Everything is resumable: finished caches, models, generations and scores are skipped.

Paper -> experiment ids: table1 table2 table3 table4 table5 table6 figure3 table8 (Table 7 is the human study).
"""
import argparse
import json
import os
import subprocess
import sys

from .handoff import HANDOFF_EXIT, should_stop, stop_now
from .common import RUNS

NUM_LAYERS = {"llama2-7b": 32, "mistral-7b": 32, "gemma-7b": 28, "deepseek-7b": 30, "llama2-7b-chat": 32,
              "mistral-7b-instruct": 32, "gemma-7b-it": 28, "deepseek-7b-chat": 30}
# DeepSeek-LLM-7B has 30 layers.

BASE = dict(variant="constant", rank=8, lam=0.1, lr=2e-4, batch=16, seed=42, scheme="parallel",
            b_init="normal", freeze_b=False, wd=0.01, data_tag="", layer=None, mix=0.0)


def R(**kw):
    r = dict(BASE)
    r.update(kw)
    return r


def run_name(r):
    if r.get("base"):
        return "base"
    if r.get("method"):  # matched baselines (ambs/baselines.py)
        return f"{r['method']}-s{r.get('seed', 42)}"
    if r["layer"] is not None:
        if r["variant"] == "constant" and r["scheme"] == "parallel":
            return f"constant-r{r['rank']}-lam{r['lam']}-lr{r['lr']}-b{r['batch']}-layer{r['layer']}-s{r['seed']}"
        return (f"{r['variant']}-r{r['rank']}-lam{r['lam']}-lr{r['lr']}-b{r['batch']}"
                + ("-pooled" if r["scheme"] == "pooled" else "") + f"-layer{r['layer']}-s{r['seed']}")
    parts = [r["variant"], f"r{r['rank']}", f"lam{r['lam']}", f"lr{r['lr']}", f"b{r['batch']}"]
    if r["scheme"] != "parallel":
        parts.append(r["scheme"])
    if r["b_init"] != "normal":
        parts.append(f"binit-{r['b_init']}")
    if r["freeze_b"]:
        parts.append("freezeb")
    if r["wd"] != 0.01:
        parts.append(f"wd{r['wd']}")
    if r["data_tag"]:
        parts.append(r["data_tag"])
    if r.get("mix"):
        parts.append(f"mix{r['mix']}")
    parts.append(f"s{r['seed']}")
    return "-".join(parts)


def plan(exp, bb):
    """List of (label, run-spec, modes, benches)."""
    L = NUM_LAYERS[bb]
    three = "alpacaeval,truthfulqa,beavertails"
    C42, IC42 = R(), R(variant="ic")
    DV = R(variant="direct_vector", wd=0.0)  # paper: parameter-space weight decay disabled for the matched control
    if exp == "interference":  # pathway x benchmark matrix; lambda = 0.1 is the Table 1 run
        paths = ["path-help", "path-harm", "path-honest"]
        return [("lambda=0", R(lam=0.0), paths, three), ("lambda=1.0", R(lam=1.0), paths, three)]
    if exp == "targets":  # supervision-target quality (ambs/targets.py)
        fam = "llama2" if bb.startswith("llama2") else "mistral"
        return [("Stronger targets", R(data_tag="strong"), ["joint"], three),
                ("Self-distilled targets", R(data_tag=f"self_{fam}"), ["joint"], three)]
    if exp == "baselines":  # matched baselines under the same protocol
        return [(m, {"method": m, "seed": 42}, ["joint"], three) for m in ("caa", "lora_sft", "dpo")]
    if exp in ("selfdistill", "efficiency", "preftest", "seeds", "ctrl", "cap", "ood", "ontopic", "deploy", "rep", "jointresp", "rep2", "rep2seeds", "ksfix"):
        return []
    if exp == "mech":  # function-class control: one rank-N*r affine offset on mixed targets, 3 seeds; constant seeds
        return [(f"Single affine rank 24 s{s_}", R(variant="ic", rank=24, scheme="pooled", seed=s_), ["joint"], three)
                for s_ in (42, 43, 44)] + [(f"AMBS constant s{s_}", R(seed=s_), ["joint"], three) for s_ in (43, 44)]
    if exp == "mechmid":  # objective-specific maps vs one map of equal rank at the middle layer
        return [(f"AMBS-IC l=L/2 s{s_}", R(variant="ic", layer=L // 2, seed=s_), ["joint"], three) for s_ in (42, 43, 44)] + \
               [(f"Single affine rank 24 l=L/2 s{s_}", R(variant="ic", rank=24, scheme="pooled", layer=L // 2, seed=s_),
                 ["joint"], three) for s_ in (42, 43, 44)]
    if exp == "mechfull":  # the Table 14 outputs scored on the full test sets (judging only)
        return plan("mech", bb) + [(f"AMBS-IC s{s_}", R(variant="ic", seed=s_), ["joint"], three) for s_ in (42, 43, 44)]
    S3 = (42, 43, 44)
    if exp == "e2":  # AMBS-IC with the pooled decoder's mixture loss added (alpha = 1)
        return [(f"AMBS-IC + J_mix s{s_}", R(variant="ic", mix=1.0, seed=s_), ["joint"], three) for s_ in S3]
    if exp == "e2w":  # mixture-loss weight sweep (alpha = 0.5 and 2; alpha = 1 is e2)
        return [(f"AMBS-IC + {m} J_mix s{s_}", R(variant="ic", mix=m, seed=s_), ["joint"], three)
                for m in (0.5, 2.0) for s_ in S3]
    if exp == "e3":  # single affine map with the rank of one pathway
        return [(f"Single affine rank 8 s{s_}", R(variant="ic", rank=8, scheme="pooled", seed=s_), ["joint"], three)
                for s_ in S3]
    if exp == "depthval":  # constant offsets at l in {1, L/4, L/2, 3L/4, L}; l* chosen on validation loss
        return [(f"AMBS l={l_}", R(layer=l_) if l_ < L else R(), ["joint"], three)
                for l_ in sorted({1, L // 4, L // 2, (3 * L) // 4, L})]
    if exp == "harmaffine":  # the single affine control on HarmBench
        return [("Single affine rank 24", R(variant="ic", rank=24, scheme="pooled"), ["joint"], "harmbench")]
    if exp == "iclam":  # coupling strength vs pathway collapse (AMBS-IC, lambda 0 and 1, seed 42)
        return [(f"AMBS-IC lambda={v}", R(variant="ic", lam=v), ["joint"], three) for v in (0.0, 1.0)]
    if exp in ("lg3", "gradconf"):  # Llama Guard 3 on the decomposition outputs / gradient conflict
        return []
    if exp == "heldout":  # held-out and validation loss (no generation, no judge)
        return []
    if exp == "interf0":  # pathway x benchmark matrix without coupling (other backbones; LLaMA-2 is in interference)
        return [("lambda=0", R(lam=0.0), ["path-help", "path-harm", "path-honest"], three)]
    if exp == "mid_joint":  # intermediate-layer offsets (Table 6) under the main joint-decoding protocol
        return [("AMBS l=1 (A_joint)", R(layer=1), ["joint"], three),
                ("AMBS l=L/2 (A_joint)", R(layer=L // 2), ["joint"], three)]
    if exp == "table1":
        return [("Base", {"base": True}, ["base"], three),
                ("AMBS (constant)", C42, ["path-help", "path-harm", "path-honest", "joint"], three),
                ("AMBS-IC", IC42, ["joint"], three)]
    if exp == "table2":
        return [(f"AMBS seed {s}", R(seed=s), ["sep"], three) for s in (43, 44)] + \
               [(f"AMBS-IC seed {s}", R(variant="ic", seed=s), ["joint"], three) for s in (43, 44)]
    if exp == "table3":
        return [("AMBS perturbed", R(data_tag="pert"), ["joint"], three),
                ("Direct-vector control", DV, ["joint"], three),
                ("Direct-vector perturbed", R(variant="direct_vector", wd=0.0, data_tag="pert"), ["joint"], three),
                ("Pooled single-offset control", R(variant="direct_vector", wd=0.0, scheme="pooled"), ["joint"], three)]
    if exp == "table4":  # judges / classifiers on the fixed response cache of AMBS (A_joint); no new generation
        return [("AMBS (A_joint) cache", C42, ["joint"], three)]
    if exp == "table5":
        return [("Base", {"base": True}, ["base"], "harmbench"), ("Direct-Vector", DV, ["joint"], "harmbench"),
                ("AMBS", C42, ["joint"], "harmbench"), ("AMBS-IC", IC42, ["joint"], "harmbench")]
    if exp == "table6":
        return [(lab, spec, ["sep"], three) for lab, spec in [
            ("Early Layer (l=1)", R(layer=1)),
            ("Middle Layer (l=L/2)", R(layer=L // 2)),
            ("Direct Update", R(variant="direct_update")),
            ("No Shared Reference", R(variant="no_shared_ref")),
            ("Full-Rank (r=d)", R(rank=0)),
            ("Lower Rank (r=2)", R(rank=2)),
            ("Zero-Start Bias", R(b_init="zero")),
            ("Frozen Deviation", R(freeze_b=True)),
            ("No Consistency (lambda=0)", R(lam=0.0)),
            ("High Consistency (lambda=1.0)", R(lam=1.0)),
            ("Sequential (No Parallel)", R(scheme="sequential")),
            ("Single-Path (N=1)", R(scheme="pooled")),
            ("Shared Labels", R(scheme="shared_labels")),
            ("AMBS (Constant, Full)", C42)]]
    if exp == "figure3":
        layers = sorted({int(round(1 + k * (L - 1) / 6)) for k in range(7)})
        pts = [(f"r={v}", R(rank=v)) for v in (1, 2, 4, 8, 16, 32, 64)]
        pts += [(f"lambda={v}", R(lam=v)) for v in (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)]
        pts += [(f"layer={v}", R(layer=v) if v < L else C42) for v in layers]
        pts += [(f"lr={v}", R(lr=v)) for v in (1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3)]
        pts += [(f"batch={v}", R(batch=v)) for v in (2, 4, 8, 16, 32, 64, 128)]
        return [(lab, spec, ["sep"], three) for lab, spec in pts]
    if exp == "table8":
        return [(f"Logit-Bias seed {s}", R(variant="logit_bias", lam=0.0, seed=s), ["sep"], three) for s in (42, 43, 44)] + \
               [(f"AMBS seed {s}", R(seed=s), ["sep"], three) for s in (42, 43, 44)]
    raise ValueError(exp)


def sh(cmd):
    """Run one pipeline step. Between steps (and inside training/generation) an A100 job yields to its H200 twin."""
    if should_stop():
        stop_now("the previous pipeline step")
    print("+", cmd, flush=True)
    rc = subprocess.run(cmd, shell=True).returncode
    if rc == HANDOFF_EXIT:
        sys.exit(HANDOFF_EXIT)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


REP_METHODS = ("caa_tuned", "iti", "loreft", "matsteer")
REP2 = ("spillover", "msrs", "ksteer")  # recent multi-objective steering methods (with matsteer)
CTRL_WEIGHTS = ("0.6,0.2,0.2", "0.2,0.6,0.2", "0.2,0.2,0.6")  # equal weights = the Table 1 joint run


def ctrl_dirs(bb):
    run = run_name(R(variant="ic"))
    return [f"{bb}/gen/{run}__w{w.replace(',', '-')}" for w in CTRL_WEIGHTS] + \
           [f"{bb}/gen/{run}__{m}" for m in ("joint", "path-help", "path-harm", "path-honest")]


def gen_tag(r, mode):
    return "base" if mode == "base" else f"{run_name(r)}__{mode}"


def backbone_dtype(bb, requested):
    """FP16 as in the paper; if a backbone overflows in FP16 (detected by ambs.cache), it is switched to BF16 once,
    for every later step, and the switch is recorded in runs/<bb>/DTYPE (reported as a deviation)."""
    f = f"{RUNS}/{bb}/DTYPE"
    return open(f).read().strip() if os.path.exists(f) else requested


def stage_gpu(exp, bb, dtype):
    # On a PBS cluster a primary job takes the stage over from a running secondary twin (ambs/handoff.py). On a local
    # machine there is no twin and this is a no-op.
    from . import handoff
    os.environ.setdefault("EXP", exp); os.environ.setdefault("BB", bb)
    if handoff.ROLE != "a100":
        if handoff.acquire("h200") == handoff.NOTHING_TO_DO:
            print(f"{exp} {bb}: GPU stage already finished (by the A100 twin)", flush=True)
            return
        _stage_gpu(exp, bb, dtype)
        handoff.finish()
    else:
        _stage_gpu(exp, bb, dtype)


def _stage_gpu(exp, bb, dtype):
    py = sys.executable
    items = plan(exp, bb)
    dtype = backbone_dtype(bb, dtype)
    if exp == "gradconf":
        sh(f"{py} -m ambs.gradconf --backbone {bb}")
        return
    if exp == "lg3":
        dirs = [f"{bb}/gen/{gen_tag(R(variant=v, rank=rk, scheme=sc, seed=s_), 'joint')}" for s_ in (42, 43, 44)
                for v, rk, sc in (("ic", 8, "parallel"), ("ic", 24, "pooled"))]
        sh(f"{py} -m ambs.safety --gen_dirs {','.join(dirs)} --classifier llamaguard3")
        return
    if exp == "heldout":
        sh(f"{py} -m ambs.heldout eval --backbone {bb} --split heldout")
        sh(f"{py} -m ambs.heldout eval --backbone {bb} --split val")
        return
    if exp == "mechfull":
        return  # generation already exists (mech); this experiment only re-judges on the full sets
    if exp == "selfdistill":
        sh(f"{py} -m ambs.targets selfdistill --backbone {bb} --dtype bfloat16")
        return
    if exp == "preftest":  # preference-trained AMBS (ambs/pref.py): both layers, validation selection, joint decoding
        for lt in ("mid", "final"):
            sh(f"{py} -m ambs.pref train --backbone {bb} --layer {lt} --dtype bfloat16")
        sh(f"{py} -m ambs.pref valacc --backbone {bb} --dtype bfloat16")
        run = json.load(open(f"{RUNS}/{bb}/pref_valacc.json"))["run"]
        sh(f"{py} -m ambs.generate --backbone {bb} --run {run} --mode joint --dtype bfloat16")
        sh(f"{py} -m ambs.safety --gen_dirs {bb}/gen/{run}__joint")
        return
    if exp == "seeds":  # matched baselines, training seeds 43 and 44
        for m in ("caa", "lora_sft", "dpo"):
            for sd in (43, 44):
                name = f"{m}-s{sd}"
                sh(f"{py} -m ambs.baselines --method {m} --backbone {bb} --out {RUNS}/{bb}/{name} --dtype bfloat16 --seed {sd}")
                extra = f"--steer {RUNS}/{bb}/{name}/steer.pt" if m == "caa" else f"--adapter {RUNS}/{bb}/{name}"
                sh(f"{py} -m ambs.generate --backbone {bb} --mode base --tag {name}__joint {extra} --dtype {dtype}")
        sh(f"{py} -m ambs.safety --gen_dirs " + ",".join(f"{bb}/gen/{m}-s{sd}__joint" for m in ("caa", "lora_sft", "dpo") for sd in (43, 44)))
        return
    if exp == "ctrl":  # decoding-time re-weighting of the trained AMBS-IC model (no retraining)
        run = run_name(R(variant="ic"))
        for wt in CTRL_WEIGHTS:
            sh(f"{py} -m ambs.generate --backbone {bb} --run {run} --mode joint --weights {wt} "
               f"--tag {run}__w{wt.replace(',', '-')} --dtype {dtype}")
        for m in ("path-help", "path-harm", "path-honest"):
            sh(f"{py} -m ambs.generate --backbone {bb} --run {run} --mode {m} --dtype {dtype}")
        sh(f"{py} -m ambs.safety --gen_dirs " + ",".join(ctrl_dirs(bb)))
        return
    if exp == "cap":
        for sy in ("base", "lora_sft", "dpo", "ambs", "ambs_ic"):
            sh(f"{py} -m ambs.evals capability --backbone {bb} --system {sy}")
        return
    if exp == "ood":
        for sy in ("base", "caa", "lora_sft", "dpo", "ambs_ic"):
            sh(f"{py} -m ambs.evals oodgen --backbone {bb} --system {sy}")
        return
    if exp == "deploy":
        sh(f"{py} -m ambs.evals deploy --backbone {bb}")
        return
    if exp == "rep":  # representation-intervention baselines (ambs/repbase.py)
        for m in REP_METHODS:
            sh(f"{py} -m ambs.repbase train --backbone {bb} --method {m} --dtype bfloat16")
            sh(f"{py} -m ambs.repbase generate --backbone {bb} --method {m} --dtype {dtype}")
        sh(f"{py} -m ambs.safety --gen_dirs " + ",".join(f"{bb}/gen/{m}-s42__joint" for m in REP_METHODS))
        return
    if exp == "rep2":  # recent steering methods on the three benchmarks and HarmBench; HarmBench for the refs
        for m in REP2:
            sh(f"{py} -m ambs.repbase train --backbone {bb} --method {m} --dtype bfloat16")
            sh(f"{py} -m ambs.repbase generate --backbone {bb} --method {m} --dtype {dtype} "
               f"--benches alpacaeval,truthfulqa,beavertails,harmbench")
        sh(f"{py} -m ambs.repbase generate --backbone {bb} --method matsteer --dtype {dtype} --benches harmbench")
        for m in ("lora_sft", "dpo"):
            sh(f"{py} -m ambs.generate --backbone {bb} --mode base --tag {m}-s42__joint --adapter {RUNS}/{bb}/{m}-s42 "
               f"--benches harmbench --dtype {dtype}")
        sh(f"{py} -m ambs.safety --gen_dirs " + ",".join(f"{bb}/gen/{m}-s42__joint" for m in REP2))
        sh(f"{py} -m ambs.harmbench --gen_dirs " + ",".join(f"{bb}/gen/{m}-s42__joint" for m in
                                                           ("matsteer",) + REP2 + ("lora_sft", "dpo")))
        return
    if exp == "ksfix":  # K-Steering seed 42 with the paper's step-size rule (largest coherent alpha)
        sh(f"{py} -m ambs.repbase train --backbone {bb} --method ksteer --dtype bfloat16")
        sh(f"{py} -m ambs.repbase generate --backbone {bb} --method ksteer --dtype {dtype} "
           f"--benches alpacaeval,truthfulqa,beavertails,harmbench")
        sh(f"{py} -m ambs.safety --gen_dirs {bb}/gen/ksteer-s42__joint")
        sh(f"{py} -m ambs.harmbench --gen_dirs {bb}/gen/ksteer-s42__joint")
        return
    if exp == "rep2seeds":  # training seeds 43 and 44 for the recent steering methods
        for s_ in (43, 44):
            for m in ("matsteer",) + REP2:
                sh(f"{py} -m ambs.repbase train --backbone {bb} --method {m} --dtype bfloat16 --seed {s_}")
                sh(f"{py} -m ambs.repbase generate --backbone {bb} --method {m} --dtype {dtype} --seed {s_}")
        sh(f"{py} -m ambs.safety --gen_dirs " + ",".join(f"{bb}/gen/{m}-s{s_}__joint" for s_ in (43, 44)
                                                         for m in ("matsteer",) + REP2))
        return
    if exp == "jointresp":
        sh(f"{py} -m ambs.evals jointresp_gen --backbone {bb}")
        return
    if exp == "efficiency":
        sh(f"{py} -m ambs.extra bench --backbone {bb}")
        sh(f"{py} -m ambs.extra efficiency --backbone {bb}")
        return
    if exp == "baselines":
        for lab, r, modes, benches in items:
            name = run_name(r)
            sh(f"{py} -m ambs.baselines --method {r['method']} --backbone {bb} --out {RUNS}/{bb}/{name} --dtype bfloat16")
            extra = f"--steer {RUNS}/{bb}/{name}/steer.pt" if r["method"] == "caa" else f"--adapter {RUNS}/{bb}/{name}"
            sh(f"{py} -m ambs.generate --backbone {bb} --mode base --tag {gen_tag(r, 'joint')} {extra} "
               f"--benches {benches} --dtype {dtype}")
        own = ",".join(f"{bb}/gen/{gen_tag(r, 'joint')}" for _, r, _, _ in items)
        sh(f"{py} -m ambs.safety --gen_dirs {own}")
        return
    tags = {r.get("data_tag", "") for _, r, _, _ in items if not r.get("base")}
    for t in sorted(tags):  # hidden-state caches (standard and, for Table 3, perturbed data)
        cmd = f"{py} -m ambs.cache --backbone {bb}" + (f" --data_tag {t}" if t else "")
        try:
            sh(f"{cmd} --dtype {dtype}")
        except subprocess.CalledProcessError:
            if dtype != "float16" or os.path.exists(f"{RUNS}/{bb}/cache/meta.json"):
                raise
            print(f"FP16 failed for {bb}; switching this backbone to BF16 (recorded in runs/{bb}/DTYPE)", flush=True)
            os.makedirs(f"{RUNS}/{bb}", exist_ok=True)
            open(f"{RUNS}/{bb}/DTYPE", "w").write("bfloat16\n")
            dtype = "bfloat16"
            sh(f"{cmd} --dtype {dtype}")
    for lab, r, modes, benches in items:
        name = run_name(r)
        if not r.get("base") and not os.path.exists(f"{RUNS}/{bb}/{name}/offsets.pt"):
            if r["layer"] is not None:
                sh(f"{py} -m ambs.train_layer --backbone {bb} --layer {r['layer']} --rank {r['rank']} --lam {r['lam']} "
                   f"--lr {r['lr']} --batch {r['batch']} --seed {r['seed']} --wd {r['wd']} --dtype {dtype} "
                   f"--variant {r['variant']} --scheme {r['scheme']}")
            else:
                cmd = (f"{py} -m ambs.train --backbone {bb} --variant {r['variant']} --scheme {r['scheme']} "
                       f"--rank {r['rank']} --lam {r['lam']} --lr {r['lr']} --batch {r['batch']} --seed {r['seed']} "
                       f"--wd {r['wd']} --b_init {r['b_init']} --name {name}")
                cmd += " --freeze_b" if r["freeze_b"] else ""
                cmd += f" --data_tag {r['data_tag']}" if r["data_tag"] else ""
                cmd += f" --mix {r['mix']}" if r.get("mix") else ""
                sh(cmd)
        for m in modes:
            run_arg = "" if m == "base" else f"--run {name}"
            sh(f"{py} -m ambs.generate --backbone {bb} {run_arg} --mode {m} --benches {benches} --dtype {dtype}")
    # score only this experiment's own output folders (another stream of the same backbone may be running)
    own = ",".join(sorted({f"{bb}/gen/{gen_tag(r, m)}" for _, r, modes, _ in items for m in modes}))
    if exp in ("table5", "harmaffine"):
        sh(f"{py} -m ambs.harmbench --gen_dirs {own}")
    else:
        sh(f"{py} -m ambs.safety --gen_dirs {own}")
    if exp == "table4":
        sh(f"{py} -m ambs.safety --gen_dirs {bb}/gen/{gen_tag(R(), 'joint')} --classifier llamaguard3")


def stage_judge(exp, bb, batch):
    py = sys.executable
    if exp in ("table5", "harmaffine", "heldout", "lg3", "gradconf"):
        return  # local classifier / no judge
    dirs = sorted({f"{bb}/gen/{gen_tag(r, m)}" for _, r, modes, _ in plan(exp, bb) for m in modes})
    if exp == "preftest":
        dirs = [f"{bb}/gen/{json.load(open(f'{RUNS}/{bb}/pref_valacc.json'))['run']}__joint"]
    if exp == "jointresp":
        sh(f"{py} -m ambs.evals jointresp_judge --backbone {bb}")
        return
    if exp == "rep":
        sh(f"{py} -m ambs.judge --gen_dirs " + ",".join([f"{bb}/gen/{m}-s42__joint" for m in REP_METHODS] +
                                                         [f"{bb}/gen/{gen_tag(R(variant='ic'), 'joint')}"]) + " --subset 200"
           + (" --batch" if batch else ""))
        return
    if exp == "rep2":  # full test sets (Table 1), then the 200-prompt subsets from the cache, then on-topic
        dirs = ",".join(f"{bb}/gen/{m}-s42__joint" for m in ("matsteer",) + REP2)
        sh(f"{py} -m ambs.judge --gen_dirs {dirs}" + (" --batch" if batch else ""))
        sh(f"{py} -m ambs.judge --gen_dirs {dirs} --subset 200")
        for sy in ("affine", "lora_sft", "matsteer") + REP2:
            sh(f"{py} -m ambs.evals ontopic --backbone {bb} --system {sy}")
        return
    if exp == "ksfix":
        d = f"{bb}/gen/ksteer-s42__joint"
        sh(f"{py} -m ambs.judge --gen_dirs {d}" + (" --batch" if batch else ""))
        sh(f"{py} -m ambs.judge --gen_dirs {d} --subset 200")
        sh(f"{py} -m ambs.evals ontopic --backbone {bb} --system ksteer")
        return
    if exp == "rep2seeds":
        sh(f"{py} -m ambs.judge --gen_dirs " + ",".join(f"{bb}/gen/{m}-s{s_}__joint" for s_ in (43, 44)
                                                        for m in ("matsteer",) + REP2) + " --subset 200"
           + (" --batch" if batch else ""))
        return
    if exp in ("ood", "ontopic"):
        for sy in (("base", "caa", "lora_sft", "dpo", "ambs_ic") if exp == "ood" else ("base", "dpo", "ambs", "ambs_ic")):
            sh(f"{py} -m ambs.evals {'oodjudge' if exp == 'ood' else 'ontopic'} --backbone {bb} --system {sy}")
        return
    if exp == "mech":  # existing rows are re-scored on the same subset from the cache (no new API calls)
        dirs += [f"{bb}/gen/base", f"{bb}/gen/{gen_tag(R(), 'joint')}"] + \
                [f"{bb}/gen/{gen_tag(R(variant='ic', seed=s_), 'joint')}" for s_ in (42, 43, 44)]
    if exp == "mechfull":
        dirs += [f"{bb}/gen/base", f"{bb}/gen/{gen_tag(R(), 'joint')}"]
    if exp == "seeds":
        dirs = [f"{bb}/gen/{m}-s{sd}__joint" for m in ("caa", "lora_sft", "dpo") for sd in (42, 43, 44)]
    if exp == "ctrl":
        dirs = ctrl_dirs(bb)
    if exp in ("mech", "mechmid", "seeds", "ctrl", "e2", "e2w", "e3", "depthval", "iclam"):  # budget: fixed 200-prompt subsets
        sh(f"{py} -m ambs.judge --gen_dirs {','.join(sorted(set(dirs)))} --subset 200" + (" --batch" if batch else ""))
        return
    flag = " --batch" if batch else ""
    if exp == "figure3":  # budget: fixed 200-prompt subsets (seed 42) for AlpacaEval and TruthfulQA; safety stays full
        flag += " --subset 200"
    sh(f"{py} -m ambs.judge --gen_dirs {','.join(dirs)}{flag}")
    if exp == "table4":
        for p in (1, 2):
            sh(f"{py} -m ambs.judge --gen_dirs {','.join(dirs)} --prompt {p}{flag}")
        if os.path.exists(os.path.join(os.path.dirname(RUNS), "secrets", "anthropic_key")):
            for p in (0, 1, 2):
                sh(f"{py} -m ambs.judge --gen_dirs {','.join(dirs)} --judge claude --prompt {p}")
        else:
            print("no anthropic_key: skipping the Claude judge", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--stage", required=True, choices=["gpu", "judge", "list"])
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--realtime", action="store_true", help="judge with the realtime API instead of the Batch API")
    a = ap.parse_args()
    if a.stage == "list":
        for lab, r, modes, benches in plan(a.exp, a.backbone):
            print(f"{lab:32s} {run_name(r):60s} {','.join(modes):40s} {benches}")
    elif a.stage == "gpu":
        stage_gpu(a.exp, a.backbone, a.dtype)
    else:
        stage_judge(a.exp, a.backbone, batch=not a.realtime)


if __name__ == "__main__":
    main()
