"""Constant-offset AMBS with the intervention at decoder layer l < L (Table 6 Early/Middle Layer; Fig. 3 layer sweep).

h~_l = h_l + c_i is added to the output of block l (Eq. 4 at layer l); the remaining frozen blocks, final norm and
W_o then produce the logits, so gradients flow back through layers l+1..L. Hidden-state caching does not apply.
Same data, loss (Eq. 12), batching, optimizer and schedule as ambs.train. FP16 backbone with loss scaling.
l = L is the paper's default (after the final norm) and is trained by ambs.train instead.
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .cache import tokenize_example
from .common import ALL_BACKBONES as BACKBONES, DATA, PATHWAYS, RUNS, read_jsonl, seed_all, write_json
from .model import LayerHook, Offsets
from .handoff import should_stop, stop_now
from .train import STOP_AFTER, Sampler, save_ckpt

PEN_TOKENS = 512  # layer-l states per dataset used for the input-conditioned consistency penalty
CKPT_EVERY = int(os.environ.get("AMBS_CKPT_EVERY_LAYER", "200"))  # layer runs take ~1 h; checkpoint every ~3-4 min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--variant", default="constant", choices=["constant", "ic"],
                    help="ic: input-conditioned offsets c_i(h) = h A_i B_i + b_i on the layer-l hidden state")
    ap.add_argument("--scheme", default="parallel", choices=["parallel", "pooled"],
                    help="pooled: one pathway trained on all datasets with equal weight (single-map control)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"],
                    help="backbone precision; bfloat16 for backbones that overflow in FP16 (runs/<bb>/DTYPE)")
    a = ap.parse_args()
    seed_all(a.seed)
    if a.variant == "constant" and a.scheme == "parallel":
        name = f"constant-r{a.rank}-lam{a.lam}-lr{a.lr}-b{a.batch}-layer{a.layer}-s{a.seed}"
    else:  # must match ambs.experiments.run_name
        name = (f"{a.variant}-r{a.rank}-lam{a.lam}-lr{a.lr}-b{a.batch}" + ("-pooled" if a.scheme == "pooled" else "")
                + f"-layer{a.layer}-s{a.seed}")
    n_path = 1 if a.scheme == "pooled" else len(PATHWAYS)
    out = f"{RUNS}/{a.backbone}/{name}"
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/offsets.pt"):
        print("already trained:", out); return
    repo = BACKBONES.get(a.backbone, a.backbone)  # a local path is accepted (tests)
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # CPU only for tests
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=getattr(torch, a.dtype) if dev == "cuda" else torch.float32).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    L = model.config.num_hidden_layers
    assert 1 <= a.layer < L, f"layer must be in [1, {L-1}] (l = L is the default, trained by ambs.train)"
    d, vocab = model.config.hidden_size, model.get_output_embeddings().weight.shape[0]
    mod = Offsets(d, vocab, r=a.rank, n=n_path, variant=a.variant, seed=a.seed).to(dev)

    data = {}
    for pw in PATHWAYS:
        ex = [tokenize_example(tok, r, a.backbone) for r in read_jsonl(f"{DATA}/train_{pw}.jsonl")]
        data[pw] = [(ids, plen) for ids, plen, n in ex if n > 0]
    n_ex = {pw: len(v) for pw, v in data.items()}
    samplers = {pw: Sampler(n_ex[pw], a.seed + 1000 * k) for k, pw in enumerate(PATHWAYS)}
    spe = math.ceil(max(n_ex.values()) / a.batch)
    total = spe * a.epochs if not a.max_steps else min(spe * a.epochs, a.max_steps)
    cfg = dict(vars(a), name=name, variant=a.variant, n_pathways=n_path, d=d, vocab=vocab,
               n_examples=n_ex, steps_per_epoch=spe, total_steps=total, num_layers=L)
    write_json(f"{out}/config.json", cfg)
    print(json.dumps(cfg), flush=True)

    opt = torch.optim.AdamW(mod.parameters(), lr=a.lr, betas=(0.9, 0.999), weight_decay=a.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda" and a.dtype == "float16")  # BF16 needs no loss scaling
    block = model.model.layers[a.layer - 1]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    def pathway_loss(i, pw):
        items = [data[pw][k] for k in samplers[pw].take(a.batch)]
        T = max(len(x) for x, _ in items)
        ids = torch.full((len(items), T), pad, dtype=torch.long)
        att = torch.zeros((len(items), T), dtype=torch.long)
        lab = torch.full((len(items), T), -100, dtype=torch.long)
        for j, (x, plen) in enumerate(items):
            ids[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
            lab[j, plen:len(x)] = torch.tensor(x[plen:])
        w = [1.0 if k == i else 0.0 for k in range(n_path)]
        cap = {}

        def hook(module, inputs, output):  # keep the un-steered layer-l states for the input-conditioned penalty
            cap["h"] = (output[0] if isinstance(output, tuple) else output).detach()
            return LayerHook(mod, w)(module, inputs, output)
        hdl = block.register_forward_hook(hook)
        try:
            logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits.float()
        finally:
            hdl.remove()
        lab = lab.to(dev)
        h_valid = cap["h"][att.to(dev).bool()].float()[:PEN_TOKENS]
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1),
                               ignore_index=-100, reduction="sum") / a.batch, h_valid

    ckpt_path = f"{out}/checkpoint.pt"
    start = 1
    t0, run, n_log = time.time(), {}, 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        mod.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        for pw in PATHWAYS:
            samplers[pw].load(ck["samplers"][pw])
        run, n_log, start = ck["run"], ck["n_log"], ck["step"] + 1
        keep = [l for l in open(f"{out}/train_log.jsonl") if json.loads(l)["step"] <= ck["step"]]
        open(f"{out}/train_log.jsonl", "w").writelines(keep)
        print(f"resuming from checkpoint at step {ck['step']}", flush=True)
    log = open(f"{out}/train_log.jsonl", "a" if start > 1 else "w")
    for step in range(start, total + 1):
        opt.zero_grad(set_to_none=True)
        hs_pen = []
        for i, pw in enumerate(PATHWAYS):  # pooled: the single pathway 0 sees one batch from every dataset
            nll, h_valid = pathway_loss(0 if a.scheme == "pooled" else i, pw)
            scaler.scale(nll).backward()
            run[f"nll_{pw}"] = run.get(f"nll_{pw}", 0.0) + nll.item()
            hs_pen.append(h_valid)
        # constant offsets: the penalty does not depend on h; ic: token-averaged over the batch's layer-l states
        pen = mod.penalty(torch.cat(hs_pen) if a.variant == "ic" else torch.zeros(1, d, device=dev))
        if mod.n > 1:  # a single pathway has no pairwise penalty
            scaler.scale(a.lam * pen).backward()
        run["penalty"] = run.get("penalty", 0.0) + pen.item()
        scaler.step(opt); scaler.update()
        n_log += 1
        if step % 50 == 0 or step == total:
            rec = {"step": step, "sec": round(time.time() - t0), **{k: v / n_log for k, v in run.items()}}
            if any(isinstance(v, float) and not math.isfinite(v) for v in rec.values()):
                raise RuntimeError(f"non-finite loss: {rec}")
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
            run, n_log = {}, 0
        if step % CKPT_EVERY == 0 and step < total:
            save_ckpt(ckpt_path, {"model": mod.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                                  "step": step, "run": run, "n_log": n_log,
                                  "samplers": {pw: samplers[pw].state() for pw in PATHWAYS}})
        if STOP_AFTER and step >= STOP_AFTER:
            raise SystemExit(f"simulated kill at step {step}")
        if step % 50 == 0 and step < total and should_stop():  # A100 -> H200 handoff (ambs/handoff.py)
            save_ckpt(ckpt_path, {"model": mod.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                                  "step": step, "run": run, "n_log": n_log,
                                  "samplers": {pw: samplers[pw].state() for pw in PATHWAYS}})
            stop_now(f"layer training step {step}")
    torch.save({k: v.detach().cpu() for k, v in mod.state_dict().items()}, f"{out}/offsets.pt")
    write_json(f"{out}/done.json", {"seconds": round(time.time() - t0), "steps": total})
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
