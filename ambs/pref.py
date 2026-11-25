"""Preference-trained AMBS (AMBS-P): per-objective input-conditioned offsets trained with the DPO loss.

  python -m ambs.pref train  --backbone llama2-7b --layer mid      # offsets after decoder block L/2
  python -m ambs.pref train  --backbone llama2-7b --layer final    # offsets on the final readout (l = L)
  python -m ambs.pref valacc --backbone llama2-7b                  # held-out preference accuracy, selects the layer

Model. The backbone (including W_o) stays frozen. Pathway i adds c_i(h) = h A_i B_i + b_i (rank 8, the AMBS-IC form in
ambs/model.py) to the residual stream after block l (or to the final readout), so the offsets act on every later
layer. Decoding uses the pooled offset sum_i w_i c_i(h); because c_i is linear in h this is one forward pass, and the
weights w can be changed after training (sep / joint / path-* modes of ambs.generate read the saved config).

Training. Same preference pairs, beta, pair batch, step count and objective rotation as the DPO baseline
(ambs/baselines.py): data/pairs_{help,harm,honest}.jsonl, beta 0.1, 8 pairs per step, 1,000 steps, objectives in turn.
For a step on objective i only pathway i is active; the reference policy is the frozen backbone with no offset, which
is exactly the policy at initialisation (A ~ N(0, 1/d), B = 0, b = 0, so c_i = 0). The AMBS pairwise consistency
penalty lambda * sum_{i<j} ||c_i(h) - c_j(h)||^2 (token mean, lambda 0.1) is added on the layer-l states of the batch.
AdamW lr 2e-4, weight decay 0.01 (the AMBS defaults), gradient clipping 1.0. 10% of each pair file (seed 42) is held out
for validation and never trained on; `valacc` selects the layer by mean held-out preference accuracy (tie -> mid).
Checkpoints every 100 steps; yields to an H200 twin at a checkpoint (ambs/handoff.py); batches come from a per-step
random stream, so a resumed run trains on exactly the batches of an uninterrupted one.
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from .baselines import batchify, encode, load
from .common import DATA, PATHWAYS, RUNS, read_jsonl, seed_all
from .handoff import should_stop, stop_now
from .model import Offsets, final_norm, install_hook

DEV = os.environ.get("AMBS_DEVICE", "cuda")
STEPS = int(os.environ.get("AMBS_PREF_STEPS", 1000))
BETA, PAIRS, LR, WD, RANK, LAM, SEED, VAL_FRAC = 0.1, 8, 2e-4, 0.01, 8, 0.1, 42, 0.1


def split_pairs(o):
    """Deterministic 90/10 train/validation split of data/pairs_<o>.jsonl (seed 42)."""
    rows = read_jsonl(f"{DATA}/pairs_{o}.jsonl")
    idx = list(range(len(rows)))
    random.Random(f"{SEED}-val-{o}").shuffle(idx)
    n_val = max(1, int(round(VAL_FRAC * len(rows))))
    val = set(idx[:n_val])
    return [r for k, r in enumerate(rows) if k not in val], [r for k, r in enumerate(rows) if k in val]


def run_name(layer_tag):
    return f"pref-ic-r{RANK}-lam{LAM}-lr{LR}-layer{layer_tag}-s{SEED}"


def resolve_layer(model, tag):
    return None if tag == "final" else model.config.num_hidden_layers // 2


class Capture:
    """Records the pre-offset states at the intervention point (for the consistency penalty)."""

    def __init__(self):
        self.h = None

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        self.h = hs.detach().float()


def attach(model, mod, layer):
    """Capture hook first (pre-offset states), then the AMBS offset hook; returns (weights list, capture, handles)."""
    w = [0.0] * mod.n
    cap = Capture()
    target = model.model.layers[layer - 1] if layer is not None else final_norm(model)
    h1 = target.register_forward_hook(cap)
    h2 = install_hook(model, mod, w, layer=layer)
    return w, cap, (h1, h2)


def seq_logp(model, ids, att, msk):
    lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    lp = torch.gather(lg.log_softmax(-1), 2, ids[:, 1:, None])[..., 0]
    return (lp * msk[:, 1:]).sum(1)


def margins(bb, tok, model, w, batch, i):
    """DPO margins of pathway i on a list of pairs; returns (margin tensor, attention mask)."""
    seqs = [encode(bb, tok, r, r["chosen"]) for r in batch] + [encode(bb, tok, r, r["rejected"]) for r in batch]
    ids, att, msk = batchify(seqs, tok.pad_token_id)
    n = len(batch)
    for k in range(len(w)):
        w[k] = 0.0
    with torch.no_grad():
        ref = seq_logp(model, ids, att, msk)
    w[i] = 1.0
    pol = seq_logp(model, ids, att, msk)
    return (pol[:n] - ref[:n]) - (pol[n:] - ref[n:]), att


def train(bb, layer_tag, dtype):
    seed_all(SEED)
    tok, model = load(bb, dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layer = resolve_layer(model, layer_tag)
    d, vocab = model.config.hidden_size, model.get_output_embeddings().weight.shape[0]
    name = run_name(layer_tag)
    out = f"{RUNS}/{bb}/{name}"
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/offsets.pt"):
        print("already trained:", out); return
    mod = Offsets(d, vocab, r=RANK, n=len(PATHWAYS), variant="ic", seed=SEED, b_init="zero").to(DEV)
    params = [p for p in mod.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    ck = f"{out}/ckpt"
    state = {"step": 0, "sec": 0.0}
    if os.path.exists(f"{ck}/state.json"):
        state = json.load(open(f"{ck}/state.json"))
        mod.load_state_dict(torch.load(f"{ck}/offsets.pt", map_location=DEV))
        opt.load_state_dict(torch.load(f"{ck}/opt.pt", map_location=DEV))
    w, cap, _ = attach(model, mod, layer)
    data = {o: split_pairs(o)[0] for o in PATHWAYS}
    log = open(f"{out}/train_log.jsonl", "a")
    t0 = time.time()
    for step in range(state["step"] + 1, STEPS + 1):
        i = (step - 1) % len(PATHWAYS)
        rng = random.Random(f"{SEED}-{step}")
        batch = [rng.choice(data[PATHWAYS[i]]) for _ in range(PAIRS)]
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            m, att = margins(bb, tok, model, w, batch, i)
            dpo = -F.logsigmoid(BETA * m).mean()
            pen = mod.penalty(cap.h[att.bool()]) if LAM > 0 else torch.zeros((), device=m.device)
            loss = dpo + LAM * pen
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 25 == 0:
            log.write(json.dumps({"step": step, "objective": PATHWAYS[i], "dpo": dpo.item(), "penalty": float(pen),
                                  "acc": (m > 0).float().mean().item(), "sec": state["sec"] + time.time() - t0}) + "\n")
            log.flush()
        stop = should_stop() and step < STEPS
        if step % 100 == 0 or stop or step == STEPS:
            os.makedirs(ck, exist_ok=True)
            torch.save(mod.state_dict(), f"{ck}/offsets.pt")
            torch.save(opt.state_dict(), f"{ck}/opt.pt")
            json.dump({"step": step, "sec": state["sec"] + time.time() - t0}, open(f"{ck}/state.json", "w"))
        if stop:
            stop_now(f"pref {layer_tag} step {step}")
    torch.save(mod.state_dict(), f"{out}/offsets.pt")
    json.dump({"method": "pref", "variant": "ic", "rank": RANK, "n_pathways": len(PATHWAYS), "seed": SEED,
               "layer": layer, "layer_tag": layer_tag, "d": d, "vocab": vocab, "beta": BETA, "pairs": PAIRS,
               "steps": STEPS, "lr": LR, "wd": WD, "lam": LAM, "val_frac": VAL_FRAC,
               "trainable_params": sum(p.numel() for p in params),
               "train_sec": state["sec"] + time.time() - t0}, open(f"{out}/config.json", "w"), indent=1)


def valacc(bb, dtype, tags=("mid", "final")):
    """Held-out preference accuracy of each trained layer variant; writes valacc.json and selects the layer."""
    tok, model = load(bb, dtype)
    model.eval()
    res = {}
    for tag in tags:
        out = f"{RUNS}/{bb}/{run_name(tag)}"
        if not os.path.exists(f"{out}/offsets.pt"):
            continue
        cfg = json.load(open(f"{out}/config.json"))
        mod = Offsets(cfg["d"], cfg["vocab"], r=cfg["rank"], n=cfg["n_pathways"], variant="ic", seed=cfg["seed"]).to(DEV)
        mod.load_state_dict(torch.load(f"{out}/offsets.pt", map_location=DEV))
        w, _, hs = attach(model, mod, cfg["layer"])
        accs = {}
        with torch.no_grad():
            for i, o in enumerate(PATHWAYS):
                val = split_pairs(o)[1]
                hit = []
                for s in range(0, len(val), PAIRS):
                    m, _ = margins(bb, tok, model, w, val[s:s + PAIRS], i)
                    hit += (m > 0).tolist()
                accs[o] = 100 * sum(hit) / len(hit)
        for h in hs:
            h.remove()
        accs["mean"] = sum(accs[o] for o in PATHWAYS) / len(PATHWAYS)
        res[tag] = accs
    best = max(res, key=lambda t: (round(res[t]["mean"], 6), t == "mid"))
    out = {"valacc": res, "selected": best, "run": run_name(best)}
    json.dump(out, open(f"{RUNS}/{bb}/pref_valacc.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "valacc"])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--layer", default="mid", choices=["mid", "final"])
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    train(a.backbone, a.layer, a.dtype) if a.cmd == "train" else valacc(a.backbone, a.dtype)


if __name__ == "__main__":
    main()
