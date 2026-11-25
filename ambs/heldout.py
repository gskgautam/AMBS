"""Held-out and validation loss of trained readout/layer decoders (held-out and validation splits).

  python -m ambs.heldout prep                       # login node (needs the HF dataset cache / internet)
  python -m ambs.heldout eval --backbone B --split heldout|val [--runs r1,r2,...]

Splits (data/train_<pw>.<split>.jsonl, same row format as the training files):
  heldout  test reference responses: AlpacaEval text-davinci-003 references (805), BeaverTails 30k_test responses
           labelled safe, TruthfulQA held-out best answers (409). Used only to report loss, never to select anything.
  val      never used for training or testing: 500 unused Alpaca examples (same shuffle as training, positions
           20000-20499), 500 PKU-SafeRLHF preferred responses (prompts outside the BeaverTails test set), and for each
           TruthfulQA training question one correct answer different from its best answer. Used to select depth (E4).

For every run the deployed pooled decoder (equal pathway weights) is scored: summed response-token NLL per example,
averaged over each dataset, plus their sum J_mix (the equal-weight mixture loss of Proposition 1).
Writes runs/<backbone>/loss_<split>.json  {run: {"help":..,"harm":..,"honest":..,"mix":..,"n":{..}}}; resumable.
"""
import argparse
import json
import os
import random

import torch
import torch.nn.functional as F

from .common import ALL_BACKBONES as BACKBONES, DATA, PATHWAYS, RUNS, read_jsonl, write_json
from .cache import tokenize_example

SEED = 42


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def prep():
    from datasets import load_dataset
    from .prepare_data import norm

    # ---- heldout (test references) ----
    ae = read_jsonl(f"{DATA}/eval_alpacaeval.jsonl")
    write_jsonl(f"{DATA}/train_help.heldout.jsonl",
                [{"id": r["id"], "instruction": r["instruction"], "input": "", "output": r["reference"], "pathway": "help"}
                 for r in ae if r["reference"].strip()])
    bt_test = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    write_jsonl(f"{DATA}/train_harm.heldout.jsonl",
                [{"id": f"bt-{i}", "instruction": r["prompt"], "input": "", "output": r["response"], "pathway": "harm"}
                 for i, r in enumerate(bt_test) if r["is_safe"] and r["response"].strip()])
    tq = read_jsonl(f"{DATA}/eval_truthfulqa.jsonl")
    write_jsonl(f"{DATA}/train_honest.heldout.jsonl",
                [{"id": r["id"], "instruction": r["instruction"], "input": "", "output": r["best_answer"], "pathway": "honest"}
                 for r in tq])

    # ---- val (never trained on, never tested on) ----
    ae_instr = {norm(r["instruction"]) for r in ae}
    alp = load_dataset("tatsu-lab/alpaca", split="train")
    rows = [{"instruction": r["instruction"], "input": r["input"], "output": r["output"]}
            for r in alp if norm(r["instruction"]) not in ae_instr and r["output"].strip()]
    random.Random(SEED).shuffle(rows)  # identical to prepare_data: rows[:20000] are the training set
    train_ids = {(r["instruction"], r["input"]) for r in read_jsonl(f"{DATA}/train_help.jsonl")}
    val_help = [dict(r, id=f"alp-val-{i}", pathway="help") for i, r in enumerate(rows[20000:20500])]
    assert not any((r["instruction"], r["input"]) in train_ids for r in val_help), "Alpaca val overlaps training"
    write_jsonl(f"{DATA}/train_help.val.jsonl", val_help)
    test_prompts = {norm(r["prompt"]) for r in bt_test}
    strong = [r for r in read_jsonl(f"{DATA}/train_harm.strong.jsonl") if norm(r["instruction"]) not in test_prompts]
    random.Random(SEED).shuffle(strong)
    write_jsonl(f"{DATA}/train_harm.val.jsonl", [dict(r, pathway="harm") for r in strong[:500]])
    best = {r["instruction"]: r["output"] for r in read_jsonl(f"{DATA}/train_honest.jsonl")}
    seen, val_hon = set(), []
    for r in read_jsonl(f"{DATA}/train_honest.strong.jsonl"):
        q = r["instruction"]
        if q in best and q not in seen and r["output"].strip() != best[q].strip():
            seen.add(q)
            val_hon.append(dict(r, pathway="honest"))
    write_jsonl(f"{DATA}/train_honest.val.jsonl", val_hon)
    print({s: {pw: sum(1 for _ in open(f"{DATA}/train_{pw}.{s}.jsonl")) for pw in PATHWAYS} for s in ("heldout", "val")})


def default_runs(bb, split):
    from .experiments import NUM_LAYERS, R, run_name
    L = NUM_LAYERS[bb]
    if split == "val":  # E4: depth candidates of the constant offset, seed 42
        return [run_name(R(layer=l)) for l in (1, L // 4, L // 2, (3 * L) // 4)] + [run_name(R())]
    runs = ["base"]
    for s in (42, 43, 44):
        runs += [run_name(R(seed=s)), run_name(R(variant="ic", seed=s)),
                 run_name(R(variant="ic", rank=24, scheme="pooled", seed=s)),
                 run_name(R(variant="ic", rank=8, scheme="pooled", seed=s)),
                 run_name(R(variant="ic", mix=1.0, seed=s))]
    return runs


@torch.no_grad()
def evaluate(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .experiments import backbone_dtype
    from .generate import load_offsets
    from .model import install_hook

    bb = a.backbone
    out_path = f"{RUNS}/{bb}/loss_{a.split}.json"
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    runs = a.runs.split(",") if a.runs else default_runs(bb, a.split)
    todo = [r for r in runs if r not in res and (r == "base" or os.path.exists(f"{RUNS}/{bb}/{r}/offsets.pt"))]
    missing = [r for r in runs if r not in res and r not in todo]
    if missing:
        print("not trained yet (skipped):", missing, flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return
    repo = BACKBONES.get(bb, bb)  # a local path is accepted (tests)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = backbone_dtype(bb, "float16") if dev == "cuda" else "float32"
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=getattr(torch, dtype)).to(dev).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    data = {}
    for pw in PATHWAYS:
        ex = [tokenize_example(tok, r, bb) for r in read_jsonl(f"{DATA}/train_{pw}.{a.split}.jsonl")]
        data[pw] = sorted([(ids, plen) for ids, plen, n in ex if n > 0], key=lambda t: len(t[0]))

    def nll_sum(pw):
        """Summed response-token NLL per example under the currently installed hook, averaged over examples."""
        tot = 0.0
        for s in range(0, len(data[pw]), a.batch):
            items = data[pw][s:s + a.batch]
            T = max(len(x) for x, _ in items)
            ids = torch.full((len(items), T), pad, dtype=torch.long)
            att = torch.zeros((len(items), T), dtype=torch.long)
            lab = torch.full((len(items), T), -100, dtype=torch.long)
            for j, (x, plen) in enumerate(items):
                ids[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
                lab[j, plen:len(x)] = torch.tensor(x[plen:])
            logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits.float()
            tot += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), lab[:, 1:].to(dev).reshape(-1),
                                   ignore_index=-100, reduction="sum").item()
        return tot / len(data[pw])

    d, vocab = model.config.hidden_size, model.get_output_embeddings().weight.shape[0]
    for run in todo:
        handle = None
        if run != "base":
            mod, layer = load_offsets(f"{RUNS}/{bb}/{run}", d, vocab)
            mod = mod.to(dev)
            w = [1.0 / mod.n] * mod.n
            handle = install_hook(model, mod, w, layer=layer)
        try:
            r = {pw: nll_sum(pw) for pw in PATHWAYS}
        finally:
            if handle is not None:
                handle.remove()
        r["mix"] = sum(r[pw] for pw in PATHWAYS)
        r["n"] = {pw: len(data[pw]) for pw in PATHWAYS}
        res[run] = r
        write_json(out_path, res)
        print(run, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "eval"])
    ap.add_argument("--backbone")
    ap.add_argument("--split", default="heldout", choices=["heldout", "val"])
    ap.add_argument("--runs", default="")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    prep() if a.cmd == "prep" else evaluate(a)


if __name__ == "__main__":
    main()
