"""Generate responses for the three benchmarks with a given decoding mode.

modes:
  base   no offset (backbone only)
  sep    pathway per benchmark (help->AlpacaEval, harm->BeaverTails, honest->TruthfulQA)
  joint  mean offset over pathways (w = 1/3 each)
Sampling: temperature 1.0, top-p 0.9 (top-k disabled), max_new_tokens 256, Alpaca prompt template.
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import (ALL_BACKBONES as BACKBONES, DATA, MAX_NEW_TOKENS, PATHWAYS, RUNS, SEP_PATHWAY, build_prompt,
                     clean_generation, read_jsonl, seed_all, write_jsonl)
from .handoff import should_stop, stop_now
from .model import Offsets, install_hook

BENCHES = ["alpacaeval", "truthfulqa", "beavertails"]
MAX_NEW = {"harmbench": 512}  # HarmBench standard generation length; others use MAX_NEW_TOKENS (256)
DEV = os.environ.get("AMBS_DEVICE", "cuda")  # tests run on CPU


def load_offsets(run_dir, d, vocab):
    import json

    cfg = json.load(open(f"{run_dir}/config.json"))
    m = Offsets(cfg.get("d", d), cfg.get("vocab", vocab), r=cfg["rank"], n=cfg.get("n_pathways", len(PATHWAYS)),
                variant=cfg.get("variant", cfg.get("mode")), seed=cfg["seed"])
    m.load_state_dict(torch.load(f"{run_dir}/offsets.pt"))
    return m.to(DEV).eval(), cfg.get("layer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--run", default="", help="run name under runs/<backbone>/ (empty for base)")
    ap.add_argument("--mode", required=True, choices=["base", "sep", "joint", "path-help", "path-harm", "path-honest"])
    ap.add_argument("--benches", default=",".join(BENCHES))
    ap.add_argument("--batch", type=int, default=int(os.environ.get("AMBS_GEN_BATCH", "64")))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--adapter", default="", help="baselines: LoRA adapter dir to merge (ambs/baselines.py)")
    ap.add_argument("--steer", default="", help="baselines: activation-steering file {layer, vector, alpha}")
    ap.add_argument("--tag", default="", help="output folder name under gen/ (baselines)")
    ap.add_argument("--weights", default="", help="pooling weights w_help,w_harm,w_honest (controllability); needs --tag")
    a = ap.parse_args()

    repo = BACKBONES.get(a.backbone, a.backbone)  # a local path is accepted (tests)
    tag = a.tag or ("base" if a.mode == "base" else f"{a.run}__{a.mode}")
    out_dir = f"{RUNS}/{a.backbone}/gen/{tag}"
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(repo, padding_side="left", truncation_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = {"device_map": DEV} if DEV == "cuda" else {}
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=getattr(torch, a.dtype), **kw).to(DEV).eval()
    if a.adapter:  # LoRA-SFT / DPO baseline: merge the trained adapter into the backbone
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload().eval()
    if a.steer:  # activation-steering baseline: add alpha * v to the residual stream after block `layer`
        st = torch.load(a.steer, map_location="cpu")
        vec = (st["alpha"] * st["vector"]).to(DEV, getattr(torch, a.dtype))
        def _add(_m, _i, out):
            return (out[0] + vec,) + tuple(out[1:]) if isinstance(out, tuple) else out + vec
        model.model.layers[st["layer"] - 1].register_forward_hook(_add)
    offsets, layer = (None, None) if a.mode == "base" else load_offsets(
        f"{RUNS}/{a.backbone}/{a.run}", model.config.hidden_size, model.get_output_embeddings().weight.shape[0])

    for bench in a.benches.split(","):
        path = f"{out_dir}/{bench}.jsonl"
        if os.path.exists(path) and not a.limit:
            print("exists", path)
            continue
        if should_stop():
            stop_now(f"finishing the previous benchmark (next: {bench})")
        rows = read_jsonl(f"{DATA}/eval_{bench}.jsonl")
        if a.limit:
            rows = rows[: a.limit]
        handle = None
        if offsets is not None:
            n_pw = offsets.n
            if a.weights:  # decoding-time re-weighting of one trained model (no retraining)
                w = [float(x) for x in a.weights.split(",")]
            elif n_pw == 1:  # single-path / pooled: the one offset is used everywhere
                w = [1.0]
            elif a.mode == "joint":
                w = [1.0 / n_pw] * n_pw
            elif a.mode.startswith("path-"):
                w = [1.0 if pw == a.mode[5:] else 0.0 for pw in PATHWAYS[:n_pw]]
            else:
                w = [1.0 if pw == SEP_PATHWAY[bench] else 0.0 for pw in PATHWAYS[:n_pw]]
            handle = install_hook(model, offsets, w, layer=layer)
        # Resumable: finished samples are appended to <bench>.partial.jsonl after every batch, so an interrupted job
        # (or an A100 -> H200 handoff, ambs/handoff.py) restarts where it stopped. Each batch is seeded with
        # (seed, batch index over the remaining items) so a restart from the same point is deterministic.
        part = f"{out_dir}/{bench}.partial.jsonl" if not a.limit else None
        done = {r["id"]: r["raw_output"] for r in read_jsonl(part)} if part and os.path.exists(part) else {}
        if done:
            print(f"{bench}: resuming, {len(done)}/{len(rows)} samples already generated", flush=True)
        todo = [k for k, r in enumerate(rows) if r["id"] not in done]
        built = [build_prompt(a.backbone, tok, r["instruction"]) for r in rows]
        prompts = [b[0] for b in built]
        add_special = built[0][1] if built else True
        order = sorted(todo, key=lambda k: len(prompts[k]))
        outs = [done.get(r["id"]) for r in rows]
        t0 = time.time()
        with torch.no_grad():
            for bi, s in enumerate(range(0, len(order), a.batch)):
                seed_all(a.seed + 1000 * len(done) + bi)
                ks = order[s : s + a.batch]
                enc = tok([prompts[k] for k in ks], return_tensors="pt", padding=True, truncation=True,
                          max_length=1024, add_special_tokens=add_special).to(DEV)
                gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.9, top_k=0,
                                     repetition_penalty=1.0, max_new_tokens=MAX_NEW.get(bench, MAX_NEW_TOKENS),
                                     pad_token_id=tok.pad_token_id)
                new = []
                for j, k in enumerate(ks):
                    outs[k] = tok.decode(gen[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
                    new.append({"id": rows[k]["id"], "raw_output": outs[k]})
                if part:
                    with open(part, "a") as f:
                        for r in new:
                            f.write(json.dumps(r) + "\n")
                print(f"{bench} {len(done) + min(s + a.batch, len(order))}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
                if should_stop() and s + a.batch < len(order):
                    stop_now(f"{bench} sample {len(done) + s + len(ks)}/{len(rows)}")
        if handle is not None:
            handle.remove()
        res = [dict(id=r["id"], instruction=r["instruction"], output=clean_generation(o), raw_output=o)
               for r, o in zip(rows, outs)]
        write_jsonl(path if not a.limit else f"{out_dir}/{bench}.smoke.jsonl", res)
        if part and os.path.exists(part):
            os.remove(part)
    print("done", out_dir)


if __name__ == "__main__":
    main()
