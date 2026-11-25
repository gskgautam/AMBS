"""Train AMBS offsets (and all controls / ablations) from cached final hidden states.

Loss (Eq. 12): sum_i E_{D_i}[summed response-token NLL] + lambda * pairwise penalty (Eq. 10; IC: token-averaged).
Each step draws `batch` examples per objective; an epoch is ceil(max_i |D_i| / batch) steps; smaller sets cycle.

--scheme
  parallel       (default) N pathways trained jointly, pathway i on D_i
  sequential     Table 6 "Sequential (No Parallel)": pathways trained one after another; pathway k is coupled to the
                 already-trained, frozen pathways j<k by the same penalty
  shared_labels  Table 6 "Shared Labels": every pathway is trained on the same mixed data (one batch per dataset,
                 shared by all pathways), so there is no objective-specific supervision
  pooled         N = 1: one offset trained on all datasets with equal weight per dataset expectation (Eq. 15;
                 Table 6 "Single-Path" with --variant constant, Table 3 control C with --variant direct_vector)
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open

from .common import ALL_BACKBONES as BACKBONES, PATHWAYS, RUNS, seed_all, write_json
from .model import Offsets


def load_lm_head(repo, device):
    path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors"], local_files_only=True)
    ipath = os.path.join(path, "model.safetensors.index.json")
    if not os.path.exists(ipath):  # .bin-only checkpoints (DeepSeek): load on CPU and take the output layer
        from transformers import AutoModelForCausalLM

        m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        W = m.get_output_embeddings().weight.detach().clone()
        del m
        return W.to(device=device, dtype=torch.float32), "get_output_embeddings"
    idx = json.load(open(ipath))["weight_map"]
    key = "lm_head.weight" if "lm_head.weight" in idx else "model.embed_tokens.weight"
    with safe_open(os.path.join(path, idx[key]), framework="pt") as f:
        W = f.get_tensor(key)
    return W.to(device=device, dtype=torch.float32), key


class Sampler:
    def __init__(self, n, seed):
        self.n, self.g, self.perm, self.pos = n, np.random.default_rng(seed), None, n

    def state(self):
        return {"g": self.g.bit_generator.state, "perm": None if self.perm is None else self.perm.copy(), "pos": self.pos}

    def load(self, st):
        self.g.bit_generator.state, self.perm, self.pos = st["g"], st["perm"], st["pos"]

    def take(self, k):
        out = []
        while len(out) < k:
            if self.pos >= self.n:
                self.perm, self.pos = self.g.permutation(self.n), 0
            m = min(k - len(out), self.n - self.pos)
            out.extend(self.perm[self.pos: self.pos + m].tolist())
            self.pos += m
        return out


from .handoff import should_stop, stop_now  # noqa: E402

CKPT_EVERY = int(os.environ.get("AMBS_CKPT_EVERY", "500"))
STOP_AFTER = int(os.environ.get("AMBS_STOP_AFTER", "0"))  # tests only: simulate a job killed after N global steps


def save_ckpt(path, obj):
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run_name(a):
    if a.name:
        return a.name
    parts = [a.variant, f"r{a.rank}", f"lam{a.lam}", f"lr{a.lr}", f"b{a.batch}"]
    if a.scheme != "parallel":
        parts.append(a.scheme)
    if a.b_init != "normal":
        parts.append(f"binit-{a.b_init}")
    if a.freeze_b:
        parts.append("freezeb")
    if a.wd != 0.01:
        parts.append(f"wd{a.wd}")
    if a.data_tag:
        parts.append(a.data_tag)
    if a.mix:
        parts.append(f"mix{a.mix}")
    parts.append(f"s{a.seed}")
    return "-".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--variant", default="ic",
                    choices=["constant", "ic", "no_shared_ref", "direct_vector", "logit_bias", "direct_update"])
    ap.add_argument("--mode", dest="variant_alias", default=None, help="deprecated alias of --variant")
    ap.add_argument("--scheme", default="parallel", choices=["parallel", "sequential", "shared_labels", "pooled"])
    ap.add_argument("--rank", type=int, default=8, help="0 means full rank (r = d)")
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--b_init", default="normal", choices=["normal", "zero"])
    ap.add_argument("--freeze_b", action="store_true")
    ap.add_argument("--data_tag", default="", help="'' = standard data; 'pert' = Table 3 label-perturbed data")
    ap.add_argument("--max_steps", type=int, default=0, help="smoke tests only")
    ap.add_argument("--name", default="")
    ap.add_argument("--mix", type=float, default=0.0,
                    help="E2: add mix * J_mix(pooled decoder, equal weights) to the parallel objective")
    a = ap.parse_args()
    if a.variant_alias:
        a.variant = a.variant_alias
    assert not a.mix or (a.scheme == "parallel" and a.variant in ("ic", "constant", "direct_vector")), \
        "--mix needs the parallel scheme and a hidden-space variant"

    seed_all(a.seed)
    cache = f"{RUNS}/{a.backbone}/cache" + (f"-{a.data_tag}" if a.data_tag else "")
    meta = json.load(open(f"{cache}/meta.json"))
    d, vocab = meta["d"], meta["vocab"]
    if a.rank == 0:
        a.rank = d
    name = run_name(a)
    out = f"{RUNS}/{a.backbone}/{name}"
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/offsets.pt"):
        print("already trained:", out)
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = {}
    total_bytes = sum(os.path.getsize(f"{cache}/{pw}.h.npy") for pw in PATHWAYS)
    on_gpu = dev == "cuda" and total_bytes < 0.5 * torch.cuda.get_device_properties(0).total_memory
    for pw in PATHWAYS:
        H = torch.from_numpy(np.load(f"{cache}/{pw}.h.npy", mmap_mode="r")[:])
        H = H.to(dev) if on_gpu else (H.pin_memory() if dev == "cuda" else H)
        Y = torch.from_numpy(np.load(f"{cache}/{pw}.y.npy")).long().to(dev)
        data[pw] = (H, Y, np.load(f"{cache}/{pw}.off.npy"))
    if os.environ.get("AMBS_TEST_W"):  # unit tests only
        W, wkey = torch.load(os.environ["AMBS_TEST_W"]).float().to(dev), "test"
    else:
        W, wkey = load_lm_head(meta["backbone"], dev)
    vocab = W.shape[0]
    print(f"cache {total_bytes/1e9:.1f} GB on {'GPU' if on_gpu else 'CPU'}; lm_head {wkey} {tuple(W.shape)}", flush=True)

    n_path = 1 if a.scheme == "pooled" else len(PATHWAYS)
    mod = Offsets(d, vocab, r=a.rank, n=n_path, variant=a.variant, seed=a.seed, b_init=a.b_init,
                  freeze_b=a.freeze_b).to(dev)
    n_ex = {pw: len(data[pw][2]) - 1 for pw in PATHWAYS}
    samplers = {pw: Sampler(n_ex[pw], a.seed + 1000 * k) for k, pw in enumerate(PATHWAYS)}

    def batch_of(pw, k):
        H, Y, off = data[pw]
        ex = samplers[pw].take(k)
        idx = np.concatenate([np.arange(off[e], off[e + 1]) for e in ex])
        idx_t = torch.from_numpy(idx).to(H.device)
        return H.index_select(0, idx_t).to(dev, non_blocking=True).float(), Y.index_select(0, idx_t.to(dev))

    # Stages: list of (pathways trained in this stage, number of steps)
    spe = math.ceil(max(n_ex.values()) / a.batch)
    if a.scheme == "sequential":
        # each pathway gets the same number of updates as in parallel training (matched update budget)
        stages = [([i], spe * a.epochs) for i in range(len(PATHWAYS))]
    else:
        stages = [(list(range(n_path)), spe * a.epochs)]
    if a.max_steps:
        stages = [(p, min(s, a.max_steps)) for p, s in stages]
    cfg = dict(vars(a), name=name, n_examples=n_ex, steps_per_epoch=spe, stages=stages, d=d, vocab=vocab,
               n_pathways=n_path, n_params=sum(p.numel() for p in mod.parameters() if p.requires_grad))
    write_json(f"{out}/config.json", cfg)
    print(json.dumps(cfg), flush=True)

    ckpt_path = f"{out}/checkpoint.pt"
    resume = torch.load(ckpt_path, weights_only=False) if os.path.exists(ckpt_path) else None
    if resume:
        mod.load_state_dict(resume["model"])
        for pw in PATHWAYS:
            samplers[pw].load(resume["samplers"][pw])
        print(f"resuming from checkpoint: stage {resume['stage']} step {resume['step']}", flush=True)
    if resume:  # drop log lines written after the checkpoint (those steps are redone identically)
        keep = [l for l in open(f"{out}/train_log.jsonl") if json.loads(l)["global_step"] <= resume["global_step"]]
        open(f"{out}/train_log.jsonl", "w").writelines(keep)
    log = open(f"{out}/train_log.jsonl", "a" if resume else "w")
    t0, gstep = time.time(), (resume["global_step"] if resume else 0)
    for stage_idx, (active, steps) in enumerate(stages):
        if resume and stage_idx < resume["stage"]:
            continue
        opt = torch.optim.AdamW([p for p in mod.parameters() if p.requires_grad], lr=a.lr,
                                betas=(0.9, 0.999), weight_decay=a.wd)
        run, n_log, start = {}, 0, 1
        if resume and stage_idx == resume["stage"]:
            if resume["opt"] is not None:
                opt.load_state_dict(resume["opt"])
            run, n_log, start = resume["run"], resume["n_log"], resume["step"] + 1
            resume = None
        for step in range(start, steps + 1):
            opt.zero_grad(set_to_none=True)
            hs_all, ys_all = [], []
            if a.scheme in ("pooled", "shared_labels"):
                batches = [batch_of(pw, a.batch) for pw in PATHWAYS]  # one batch per dataset (equal weight)
                for i in active:
                    nll = sum(F.cross_entropy(mod.logits(i, h, W), y, reduction="sum") for h, y in batches) / a.batch
                    nll.backward()
                    run[f"nll_{i}"] = run.get(f"nll_{i}", 0.0) + nll.item()
                hs_all = [h for h, _ in batches]
            else:
                for i in active:
                    h, y = batch_of(PATHWAYS[i], a.batch)
                    nll = F.cross_entropy(mod.logits(i, h, W), y, reduction="sum") / a.batch
                    nll.backward()
                    run[f"nll_{PATHWAYS[i]}"] = run.get(f"nll_{PATHWAYS[i]}", 0.0) + nll.item()
                    hs_all.append(h)
                    ys_all.append(y)
                if a.mix > 0:  # E2: the deployed pooled decoder is also fitted to the equal-weight mixture
                    w = [1.0 / n_path] * n_path
                    mixl = sum(F.cross_entropy((h + mod.mixed_offset(h, w)) @ W.T, y, reduction="sum")
                               for h, y in zip(hs_all, ys_all)) / a.batch
                    (a.mix * mixl).backward()
                    run["nll_mix"] = run.get("nll_mix", 0.0) + mixl.item()
            hcat = torch.cat(hs_all)
            if a.lam > 0 and n_path > 1:
                if a.scheme == "sequential":
                    # inactive (earlier) pathways never change during this stage, so their offsets are fixed targets
                    k = active[0]
                    pen = torch.zeros((), device=dev)
                    if k > 0:
                        cur = mod.logit_delta if mod.is_logit else mod.offset
                        ck = cur(k, hcat)
                        dep = mod.variant in ("ic", "no_shared_ref", "direct_update")
                        red = (lambda x: (x ** 2).sum(-1).mean()) if dep else (lambda x: (x[0] ** 2).sum())
                        pen = sum(red(ck - cur(j, hcat).detach()) for j in range(k))
                else:
                    pen = mod.penalty(hcat)
                if pen.requires_grad:
                    (a.lam * pen).backward()
                run["penalty"] = run.get("penalty", 0.0) + float(pen)
            if a.scheme == "sequential":  # only the active pathway's slices may change
                for p in mod.parameters():
                    if p.grad is not None:
                        mask = torch.zeros_like(p.grad)
                        mask[active[0]] = 1
                        p.grad.mul_(mask)
            snap = None
            if a.scheme == "sequential":
                snap = {kk: vv.detach().clone() for kk, vv in mod.named_parameters()}
            opt.step()
            if snap is not None:  # undo AdamW decay on inactive pathway slices
                with torch.no_grad():
                    for kk, p in mod.named_parameters():
                        keep = torch.ones(p.shape[0], dtype=torch.bool, device=p.device)
                        keep[active[0]] = False
                        p[keep] = snap[kk][keep]
            n_log += 1
            gstep += 1
            if step % 50 == 0 or step == steps:
                rec = {"stage": stage_idx, "step": step, "global_step": gstep, "sec": round(time.time() - t0),
                       **{k: v / n_log for k, v in run.items()}}
                for v in rec.values():
                    if isinstance(v, float) and not math.isfinite(v):
                        raise RuntimeError(f"non-finite loss: {rec}")
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
                run, n_log = {}, 0
            if step % CKPT_EVERY == 0 and step < steps:
                save_ckpt(ckpt_path, {"model": mod.state_dict(), "opt": opt.state_dict(), "stage": stage_idx, "step": step,
                                      "global_step": gstep, "run": run, "n_log": n_log,
                                      "samplers": {pw: samplers[pw].state() for pw in PATHWAYS}})
                log.flush()
            if STOP_AFTER and gstep >= STOP_AFTER:
                raise SystemExit(f"simulated kill at global step {gstep}")
            if step % 50 == 0 and step < steps and should_stop():  # A100 -> H200 handoff (ambs/handoff.py)
                save_ckpt(ckpt_path, {"model": mod.state_dict(), "opt": opt.state_dict(), "stage": stage_idx, "step": step,
                                      "global_step": gstep, "run": run, "n_log": n_log,
                                      "samplers": {pw: samplers[pw].state() for pw in PATHWAYS}})
                stop_now(f"training step {gstep}")
        if stage_idx < len(stages) - 1:  # stage boundary checkpoint (sequential scheme)
            save_ckpt(ckpt_path, {"model": mod.state_dict(), "opt": None, "stage": stage_idx + 1, "step": 0,
                                  "global_step": gstep, "run": {}, "n_log": 0,
                                  "samplers": {pw: samplers[pw].state() for pw in PATHWAYS}})
    torch.save({k: v.detach().cpu() for k, v in mod.state_dict().items()}, f"{out}/offsets.pt")
    write_json(f"{out}/done.json", {"seconds": round(time.time() - t0), "steps": gstep})
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
