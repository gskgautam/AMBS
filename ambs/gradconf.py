"""F5: do the HHH objectives conflict at the readout?  python -m ambs.gradconf --backbone B

Cosine between per-objective gradients of the single rank-24 affine readout map c(h) = hAB + b (all of A, B, b
flattened), each objective using one training batch of 16 examples (summed response-token NLL / 16, as in training).
Measured at the seed-42 initialization and at the trained seed-42 map, over 50 batch triples.
Writes runs/<bb>/gradconf.json.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from .common import PATHWAYS, RUNS, seed_all, write_json
from .experiments import R, run_name
from .model import Offsets
from .train import Sampler, load_lm_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    seed_all(0)
    bb = a.backbone
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = f"{RUNS}/{bb}/cache"
    meta = json.load(open(f"{cache}/meta.json"))
    if os.environ.get("AMBS_TEST_W"):
        W = torch.load(os.environ["AMBS_TEST_W"]).float().to(dev)
    else:
        W, _ = load_lm_head(meta["backbone"], dev)
    data = {pw: (np.load(f"{cache}/{pw}.h.npy", mmap_mode="r"), np.load(f"{cache}/{pw}.y.npy"),
                 np.load(f"{cache}/{pw}.off.npy")) for pw in PATHWAYS}
    samplers = {pw: Sampler(len(data[pw][2]) - 1, 7 + k) for k, pw in enumerate(PATHWAYS)}
    d = meta["d"]
    run = run_name(R(variant="ic", rank=24, scheme="pooled", seed=42))
    states = {"init": Offsets(d, W.shape[0], r=24, n=1, variant="ic", seed=42).state_dict()}
    if os.path.exists(f"{RUNS}/{bb}/{run}/offsets.pt"):
        states["trained"] = torch.load(f"{RUNS}/{bb}/{run}/offsets.pt")
    batches = []
    for _ in range(a.n):
        trip = []
        for pw in PATHWAYS:
            H, Y, off = data[pw]
            idx = np.concatenate([np.arange(off[e], off[e + 1]) for e in samplers[pw].take(a.batch)])
            trip.append((torch.from_numpy(np.asarray(H[idx], dtype=np.float32)).to(dev), torch.from_numpy(Y[idx]).long().to(dev)))
        batches.append(trip)
    pairs = [(0, 1), (0, 2), (1, 2)]
    out = {}
    for tag, sd in states.items():
        mod = Offsets(d, W.shape[0], r=24, n=1, variant="ic", seed=42).to(dev)
        mod.load_state_dict(sd)
        cos = {f"{PATHWAYS[i]}-{PATHWAYS[j]}": [] for i, j in pairs}
        for trip in batches:
            g = []
            for h, y in trip:
                mod.zero_grad(set_to_none=True)
                (F.cross_entropy(mod.logits(0, h, W), y, reduction="sum") / a.batch).backward()
                g.append(torch.cat([p.grad.flatten() for p in mod.parameters() if p.grad is not None]).double())
            for i, j in pairs:
                cos[f"{PATHWAYS[i]}-{PATHWAYS[j]}"].append(float(F.cosine_similarity(g[i], g[j], dim=0)))
        out[tag] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "frac_negative": float(np.mean(np.array(v) < 0))}
                    for k, v in cos.items()}
        print(bb, tag, json.dumps({k: round(v["mean"], 3) for k, v in out[tag].items()}), flush=True)
    write_json(f"{RUNS}/{bb}/gradconf.json", out)


if __name__ == "__main__":
    main()
