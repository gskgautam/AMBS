"""Representation-intervention baselines at small parameter budgets (Table rep_baselines).

  python -m ambs.repbase train    --backbone B --method {caa_tuned,iti,loreft,matsteer}
  python -m ambs.repbase generate --backbone B --method M     # joint outputs for the three benchmarks -> gen/<M>-s42__joint

All methods use the preference pairs of the DPO baseline (data/pairs_*.jsonl) with the 90/10 train/validation split of
ambs/pref.py, pooled over the three objectives (LoReFT, like LoRA-SFT, uses the AMBS reference targets). One strength
hyperparameter per method is selected on the validation pairs by preference accuracy (log p(chosen) > log p(rejected)
under the intervened model); the test sets are never used for selection.

caa_tuned  CAA vector of ambs/baselines.py (block L/2, mean chosen-minus-rejected residual over response tokens, 512
           training pairs per objective, averaged over objectives) with multiplier selected from {0.5, 1, 2, 4, 8}.
iti        Inference-time intervention (Li et al., 2023): head outputs (input of o_proj) at the last response token;
           a logistic probe per head (chosen vs rejected, 512 training pairs per objective, 80/20 probe split); the
           K = 48 heads with the highest probe accuracy are shifted along their mass-mean direction by alpha * sigma at
           every position; alpha selected from {5, 10, 15} (15 is the paper's default).
loreft     LoReFT (Wu et al., 2024): Phi(h) = h + R^T (W h + b - R h), R in R^{8 x d} with orthonormal rows, on the
           residual stream after block L/2 at the first 7 and last 7 prompt positions; trained like LoRA-SFT (next-token
           loss on the AMBS reference targets, objectives in turn, batch 16, 1,250 steps), lr 9e-4, no weight decay.
matsteer   MAT-Steer (Nguyen et al., 2025): one steering vector theta_t and gate G_t(a) = sigmoid(w_t a + b_t) per
           objective on the self-attention output of block L/2, f(a) = a + sum_t G_t(a) theta_t rescaled to ||a||; loss =
           MMD(pos, f(neg)) (Gaussian kernel, sigma = 2) + 0.8 L_pos + 0.8 L_sparse + 0.1 L_ortho (the paper's
           generation setting), 16 positive and 16 negative samples per objective per step, Adam lr 1e-3, 1,000 steps;
           applied to prompt (query) tokens at inference. Unspecified in the paper and fixed here: per-sample features are
           the mean attention output over response tokens divided by its median norm (so sigma = 2 is meaningful),
           the optimiser and the step count.
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from .baselines import batchify, encode, load
from .common import DATA, PATHWAYS, RUNS, clean_generation, read_jsonl, seed_all, write_jsonl
from .handoff import should_stop, stop_now
from .pref import split_pairs

DEV = os.environ.get("AMBS_DEVICE", "cuda")
N_PAIRS = int(os.environ.get("AMBS_REP_PAIRS", 512))
STEPS = int(os.environ.get("AMBS_REP_STEPS", 0)) or None
SEED = 42  # set by --seed; seed 42 reproduces the original runs exactly
BENCHES = ["alpacaeval", "truthfulqa", "beavertails"]


def outdir(bb, m):
    return f"{RUNS}/{bb}/{m}-s{SEED}"


def blocks(model):
    return model.model.layers


def train_pairs(n=N_PAIRS):
    out = []
    for o in PATHWAYS:
        tr = split_pairs(o)[0]
        random.Random(SEED).shuffle(tr)
        out += [(o, p) for p in tr[:n]]
    return out


def val_pairs(n=128):
    return [(o, p) for o in PATHWAYS for p in split_pairs(o)[1][:n]]


def pref_accuracy(bb, tok, model, pairs):
    """Share of validation pairs where the intervened model gives the chosen response the higher log-probability."""
    hit = 0
    with torch.no_grad():
        for s in range(0, len(pairs), 8):
            b = [p for _, p in pairs[s:s + 8]]
            seqs = [encode(bb, tok, r, r["chosen"]) for r in b] + [encode(bb, tok, r, r["rejected"]) for r in b]
            ids, att, msk = batchify(seqs, tok.pad_token_id)
            lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            lp = (torch.gather(lg.log_softmax(-1), 2, ids[:, 1:, None])[..., 0] * msk[:, 1:]).sum(1)
            hit += int((lp[:len(b)] > lp[len(b):]).sum())
    return 100 * hit / len(pairs)


# ------------------------------------------------------------------ hooks
class AddHook:
    """Adds a fixed vector to a module output (residual stream or o_proj input via pre-hook)."""

    def __init__(self, vec):
        self.vec = vec

    def __call__(self, module, inputs, output):
        return (output[0] + self.vec,) + tuple(output[1:]) if isinstance(output, tuple) else output + self.vec


class ITIHook:
    def __init__(self, vec):
        self.vec = vec

    def __call__(self, module, args):
        return (args[0] + self.vec,) + tuple(args[1:])


class PromptPositions:
    """Shared state: which positions of the current prefill are prompt positions to intervene on (left padding)."""

    def __init__(self, first=7, last=7):
        self.first, self.last, self.mask = first, last, None

    def select(self, att):
        """att: (B, T) attention mask of the prompt -> (B, T) bool of the first/last prompt positions."""
        B, T = att.shape
        sel = torch.zeros_like(att, dtype=torch.bool)
        for i in range(B):
            idx = att[i].nonzero().flatten()
            sel[i, idx[:self.first]] = True
            sel[i, idx[-self.last:]] = True
        return sel


class LoReFT(torch.nn.Module):
    def __init__(self, d, r=8):
        super().__init__()
        self.R = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(d, r, bias=False))
        self.W = torch.nn.Linear(d, r)

    def forward(self, h):
        return h + (self.W(h) - self.R(h)) @ self.R.weight


class ReFTHook:
    def __init__(self, mod, pos):
        self.mod, self.pos = mod, pos

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self.pos.mask is not None and hs.shape[1] == self.pos.mask.shape[1]:  # prefill / training pass only
            m = self.pos.mask[..., None].to(hs.dtype)
            hs = hs * (1 - m) + self.mod(hs.float()).to(hs.dtype) * m
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs


class MATSteer(torch.nn.Module):
    def __init__(self, d, n=3, scale=1.0):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.randn(n, d) * 0.01)
        self.gate = torch.nn.Linear(d, n)
        self.scale = scale  # features are divided by `scale` (median norm) before the gate and the edit

    def forward(self, a):
        x = a / self.scale
        g = torch.sigmoid(self.gate(x))
        y = x + g @ self.theta
        y = y * (x.norm(dim=-1, keepdim=True) / y.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        return y * self.scale, g


class MATHook:
    def __init__(self, mod, pos):
        self.mod, self.pos = mod, pos

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self.pos.mask is not None and hs.shape[1] == self.pos.mask.shape[1]:  # query (prompt) tokens only
            m = self.pos.mask[..., None].to(hs.dtype)
            hs = hs * (1 - m) + self.mod(hs.float())[0].to(hs.dtype) * m
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs


# ------------------------------------------------------------------ apply a trained method
def attach(bb, model, tok, method):
    """Install the method's hook on a loaded model; returns a PromptPositions object (or None) to set per batch."""
    d = outdir(bb, method)
    L = model.config.num_hidden_layers
    if method in ("caa_tuned", "spillover"):
        st = torch.load(f"{d}/steer.pt", map_location="cpu")
        blocks(model)[st["layer"] - 1].register_forward_hook(AddHook((st["alpha"] * st["vector"]).to(DEV, model.dtype)))
        return None
    if method == "iti":
        st = torch.load(f"{d}/iti.pt", map_location="cpu")
        for layer, vec in st["shifts"].items():
            blocks(model)[layer].self_attn.o_proj.register_forward_pre_hook(ITIHook((st["alpha"] * vec).to(DEV, model.dtype)))
        return None
    if method == "ksteer":
        st = torch.load(f"{d}/ksteer.pt", map_location=DEV)
        clf = KClassifier(model.config.hidden_size, st["n_cls"]).to(DEV)
        clf.load_state_dict(st["state"])
        blocks(model)[st["layer"] - 1].register_forward_hook(KSteerHook(clf.eval(), st["alpha"], st["pos"], st["neg"]))
        return None
    if method == "msrs":
        st = torch.load(f"{d}/msrs.pt", map_location=DEV)
        mod = MSRS(model.config.hidden_size, st["rank"]).to(DEV)
        mod.load_state_dict(st["state"])
        pos = PromptPositions(first=0, last=5)
        blocks(model)[st["layer"] - 1].register_forward_hook(MSRSHook(mod.eval(), pos))
        return pos
    pos = PromptPositions() if method == "loreft" else PromptPositions(first=10 ** 6, last=0)
    if method == "loreft":
        mod = LoReFT(model.config.hidden_size).to(DEV)
        mod.load_state_dict(torch.load(f"{d}/reft.pt", map_location=DEV))
        blocks(model)[L // 2 - 1].register_forward_hook(ReFTHook(mod.eval(), pos))
    else:
        st = torch.load(f"{d}/matsteer.pt", map_location=DEV)
        mod = MATSteer(model.config.hidden_size, scale=st["scale"]).to(DEV)
        mod.load_state_dict(st["state"])
        blocks(model)[L // 2 - 1].self_attn.register_forward_hook(MATHook(mod.eval(), pos))
    return pos


# ------------------------------------------------------------------ training
def _resp_mean(model, ids, att, msk, module, kind):
    """Mean over response tokens of a module's output (kind='out') or o_proj input per head (kind='heads', last token)."""
    store = {}
    if kind == "out":
        h = module.register_forward_hook(lambda m, i, o: store.__setitem__("x", (o[0] if isinstance(o, tuple) else o)))
    else:
        h = module.register_forward_pre_hook(lambda m, a: store.__setitem__("x", a[0]))
    with torch.no_grad():
        model(input_ids=ids, attention_mask=att)
    h.remove()
    x = store["x"].float()
    m = msk.bool()
    if kind == "heads":
        last = m.long().cumsum(1).argmax(1)  # last response position
        return x[torch.arange(x.shape[0]), last]
    return torch.stack([x[i][m[i]].mean(0) if m[i].any() else x[i].mean(0) for i in range(x.shape[0])])


def train_caa_tuned(bb, dtype):
    d = outdir(bb, "caa_tuned"); os.makedirs(d, exist_ok=True)
    tok, model = load(bb, dtype)
    layer = model.config.num_hidden_layers // 2
    diffs = {o: [] for o in PATHWAYS}
    with torch.no_grad():
        for o, p in train_pairs():  # training split only, so the multiplier search on validation pairs is clean
            ms = []
            for resp in (p["chosen"], p["rejected"]):
                ids, att, msk = batchify([encode(bb, tok, p, resp)], tok.pad_token_id)
                h = model(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states[layer][0].float()
                m = msk[0].bool()
                ms.append(h[m].mean(0) if m.any() else h.mean(0))
            diffs[o].append(ms[0] - ms[1])
    vec = torch.stack([torch.stack(v).mean(0) for v in diffs.values()]).mean(0).cpu()
    vp, res = val_pairs(), {}
    for a in (0.5, 1.0, 2.0, 4.0, 8.0):
        h = blocks(model)[layer - 1].register_forward_hook(AddHook((a * vec).to(DEV, model.dtype)))
        res[a] = pref_accuracy(bb, tok, model, vp)
        h.remove()
    best = max(res, key=res.get)
    torch.save({"layer": layer, "vector": vec, "alpha": best}, f"{d}/steer.pt")
    json.dump({"method": "caa_tuned", "val_acc": res, "alpha": best, "trainable_params": vec.numel()},
              open(f"{d}/done.json", "w"), indent=1)


def train_iti(bb, dtype, K=48):
    d = outdir(bb, "iti"); os.makedirs(d, exist_ok=True)
    tok, model = load(bb, dtype)
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    dh = model.config.hidden_size // H if not hasattr(model.config, "head_dim") or model.config.head_dim is None else model.config.head_dim
    feats = {l: [] for l in range(L)}
    labels = []
    for _, p in train_pairs():
        for resp, y in ((p["chosen"], 1), (p["rejected"], 0)):
            ids, att, msk = batchify([encode(bb, tok, p, resp)], tok.pad_token_id)
            store = {}
            hs = [blocks(model)[l].self_attn.o_proj.register_forward_pre_hook(
                (lambda l: lambda m, a: store.__setitem__(l, a[0][0].float()))(l)) for l in range(L)]
            with torch.no_grad():
                model(input_ids=ids, attention_mask=att)
            for h in hs:
                h.remove()
            last = int(msk[0].nonzero().max()) if msk[0].any() else ids.shape[1] - 1
            for l in range(L):
                feats[l].append(store[l][last].cpu())
            labels.append(y)
    y = torch.tensor(labels, dtype=torch.float)
    n = len(y); idx = list(range(n)); random.Random(42).shuffle(idx)
    tr, va = idx[:int(0.8 * n)], idx[int(0.8 * n):]
    scores = []
    for l in range(L):
        X = torch.stack(feats[l]).view(n, H, dh)  # (n, H, dh)
        X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
        Wp = torch.zeros(H, dh, requires_grad=True); bp = torch.zeros(H, requires_grad=True)
        opt = torch.optim.Adam([Wp, bp], lr=0.05)
        for _ in range(200):  # batched logistic regressions, one per head
            logit = (X[tr] * Wp).sum(-1) + bp
            loss = F.binary_cross_entropy_with_logits(logit, y[tr, None].expand_as(logit)) + 1e-3 * (Wp ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (((X[va] * Wp).sum(-1) + bp > 0).float() == y[va, None]).float().mean(0)
        scores += [(float(acc[h]), l, h) for h in range(H)]
    top = sorted(scores, reverse=True)[:K]
    shifts = {}
    for _, l, h in top:
        X = torch.stack(feats[l]).view(n, H, dh)[:, h]
        dvec = X[y == 1].mean(0) - X[y == 0].mean(0)
        dvec = dvec / dvec.norm()
        sigma = (X @ dvec).std()
        v = shifts.setdefault(l, torch.zeros(H * dh))
        v[h * dh:(h + 1) * dh] = sigma * dvec
    vp, res = val_pairs(), {}
    for a in (5.0, 10.0, 15.0):
        hs = [blocks(model)[l].self_attn.o_proj.register_forward_pre_hook(ITIHook((a * v).to(DEV, model.dtype)))
              for l, v in shifts.items()]
        res[a] = pref_accuracy(bb, tok, model, vp)
        for h in hs:
            h.remove()
    best = max(res, key=res.get)
    torch.save({"shifts": shifts, "alpha": best, "heads": [(l, h) for _, l, h in top]}, f"{d}/iti.pt")
    json.dump({"method": "iti", "K": K, "val_acc": res, "alpha": best, "trainable_params": K * dh,
               "top_probe_acc": [round(a, 3) for a, _, _ in top[:5]]}, open(f"{d}/done.json", "w"), indent=1)


def train_loreft(bb, dtype, steps=1250, bs=16, lr=9e-4):
    steps = STEPS or steps
    d = outdir(bb, "loreft"); os.makedirs(d, exist_ok=True)
    seed_all(SEED)
    tok, model = load(bb, dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    L = model.config.num_hidden_layers
    mod = LoReFT(model.config.hidden_size).to(DEV)
    pos = PromptPositions()
    blocks(model)[L // 2 - 1].register_forward_hook(ReFTHook(mod, pos))
    opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=0.0)
    ck = f"{d}/ckpt.pt"
    start = 0
    if os.path.exists(ck):
        c = torch.load(ck, map_location=DEV); mod.load_state_dict(c["mod"]); opt.load_state_dict(c["opt"]); start = c["step"]
    data = {o: read_jsonl(f"{DATA}/train_{o}.jsonl") for o in PATHWAYS}
    t0 = time.time()
    for step in range(start + 1, steps + 1):
        o = PATHWAYS[(step - 1) % 3]
        rng = random.Random(f"42-{step}")
        batch = [rng.choice(data[o]) for _ in range(bs)]
        enc = [encode(bb, tok, r, r["output"]) for r in batch]
        ids, att, msk = batchify(enc, tok.pad_token_id)
        prompt_att = att * (1 - msk.long())  # prompt positions = attended and not response
        pos.mask = PromptPositions().select(prompt_att)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            nll = F.cross_entropy(lg.transpose(1, 2), ids[:, 1:], reduction="none")
            loss = (nll * msk[:, 1:]).sum() / msk[:, 1:].sum().clamp(min=1)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        stop = should_stop() and step < steps
        if step % 100 == 0 or stop:
            torch.save({"mod": mod.state_dict(), "opt": opt.state_dict(), "step": step}, ck)
        if stop:
            stop_now(f"loreft step {step}")
    torch.save(mod.state_dict(), f"{d}/reft.pt")
    json.dump({"method": "loreft", "rank": 8, "layer": L // 2, "positions": "f7+l7", "steps": steps, "lr": lr,
               "trainable_params": sum(p.numel() for p in mod.parameters()), "train_sec": time.time() - t0},
              open(f"{d}/done.json", "w"), indent=1)


def mmd(x, y, sigma=2.0):
    k = lambda a, b: torch.exp(-torch.cdist(a, b) ** 2 / (2 * sigma ** 2))
    return k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()


def train_matsteer(bb, dtype, steps=1000, lr=1e-3):
    steps = STEPS or steps
    d = outdir(bb, "matsteer"); os.makedirs(d, exist_ok=True)
    seed_all(SEED); t0 = time.time()
    tok, model = load(bb, dtype)
    L = model.config.num_hidden_layers
    attn = blocks(model)[L // 2 - 1].self_attn
    pos_f, neg_f = {o: [] for o in PATHWAYS}, {o: [] for o in PATHWAYS}
    for o, p in train_pairs():
        for resp, tgt in ((p["chosen"], pos_f), (p["rejected"], neg_f)):
            ids, att, msk = batchify([encode(bb, tok, p, resp)], tok.pad_token_id)
            tgt[o].append(_resp_mean(model, ids, att, msk, attn, "out")[0].cpu())
    P = {o: torch.stack(v).to(DEV) for o, v in pos_f.items()}
    N = {o: torch.stack(v).to(DEV) for o, v in neg_f.items()}
    scale = float(torch.cat(list(P.values()) + list(N.values())).norm(dim=-1).median())
    mod = MATSteer(model.config.hidden_size, scale=scale).to(DEV)
    opt = torch.optim.Adam(mod.parameters(), lr=lr)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    for step in range(1, steps + 1):
        loss = 0.0
        for t, o in enumerate(PATHWAYS):
            ip = torch.randint(len(P[o]), (16,), generator=g); ineg = torch.randint(len(N[o]), (16,), generator=g)
            ap, an = P[o][ip], N[o][ineg]
            fn, gn = mod(an)
            _, gp = mod(ap)
            loss = loss + mmd(ap / scale, fn / scale) + 0.8 * (gp[:, t] ** 2).sum() / 16 + 0.8 * gn[:, t].abs().sum() / 16
        th = F.normalize(mod.theta, dim=-1)
        cos = th @ th.T
        loss = loss + 0.1 * (cos ** 2 - torch.diag(torch.diag(cos) ** 2)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.save({"state": mod.state_dict(), "scale": scale}, f"{d}/matsteer.pt")
    json.dump({"method": "matsteer", "layer": L // 2, "steps": steps, "lr": lr, "scale": scale,
               "trainable_params": sum(p.numel() for p in mod.parameters()), "train_sec": time.time() - t0,
               "seed": SEED}, open(f"{d}/done.json", "w"), indent=1)


TRAIN = {"caa_tuned": train_caa_tuned, "iti": train_iti, "loreft": train_loreft, "matsteer": train_matsteer}


# ------------------------------------------------------------------ recent multi-objective steering methods
# Reimplemented from the papers; choices the papers leave open are fixed here and listed per method.
#
# spillover  Spillover-aware multi-value steering (Pan et al., 2026; no public code): one unit contrastive direction per
#            objective, d_k = normalize(mean chosen - mean rejected residual after block L/2 over response tokens, 512
#            training pairs per objective); D = [d_1..d_N], G = D^T D, coefficients alpha = G^{-1} w with equal weights
#            w = 1/N, so that d_j^T (D alpha) = w_j (the defining property, checked and stored in done.json). The vector
#            D alpha is added at every position, rescaled to ratio * median ||h|| (the paper calibrates ratio ~ 0.15);
#            ratio selected from {0.05, 0.1, 0.15, 0.2, 0.3} by validation preference accuracy. Fixed here: layer L/2
#            (the paper picks the layer by probe F1) and contrastive pairs instead of persona prompts.
# msrs       MSRS (Jiang et al., 2025; no public code found): Phi(h) = h + R^T diag(m(h)) (W h + b - R h), R in
#            R^{r x d} with orthonormal rows initialised to S_align = [B_shared; B_1; ...; B_N] (shared subspace from the
#            SVD of the per-objective mean activations, private subspaces from the SVD of each objective's activations
#            after projecting out the shared and earlier subspaces; each size = smallest k reaching 60% of the singular
#            value mass, total rank capped at 8); m(h) = sigmoid(MLP(h)). Loss = next-token loss on the AMBS reference
#            targets + 0.3 ||m(h) - 1||^2 + 0.5 (1 - cos_F(R, S_align)); Adam lr 9e-4, batch 4, 3,750 steps, residual
#            stream after block L/2, last 5 prompt positions. Fixed here: MLP width 16, 500 examples per objective for
#            the subspaces, activations averaged over the last 5 positions, m_prior = 1.
# ksteer     K-Steering (Oozeer et al., 2025): MLP classifier d -> 256 -> 256 -> 2N (ReLU, cross-entropy, Adam lr 1e-3,
#            batch 32, 30 epochs) on the residual after block L/2 at the last position of chosen (class 2t) and rejected
#            (class 2t+1) responses; at inference one gradient step a' = a - alpha * grad_a L at every position, with
#            L = -mean(logits of chosen classes) + mean(logits of rejected classes) per position (the paper averages
#            over positions too, which makes the step size depend on sequence length under KV caching); alpha selected
#            as in the paper: the largest alpha whose outputs stay coherent (the paper asks a GPT judge; here the check
#            is automatic): greedy 64-token continuations of 16 validation prompts must keep their mean token NLL
#            under the unsteered base model within log(1.5) of the unsteered continuations and at least half their
#            length; grid {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8, 16}. Alpha was first chosen by validation
#            preference accuracy, which favoured large steps that produced incoherent text (superseded).
#            'largest coherent' means the largest alpha below the first incoherent one in the ascending
#            grid; an earlier rule took the max over all coherent alphas, which on Gemma skipped failures at 2 and 8 and
#            chose 16 (mostly empty outputs). The rule change alters only Gemma (16 -> 1); the other backbones'
#            choices are identical under both rules.


def _resid_resp_means(bb, tok, model, layer, p, resp):
    ids, att, msk = batchify([encode(bb, tok, p, resp)], tok.pad_token_id)
    h = model(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states[layer][0].float()
    m = msk[0].bool()
    x = h[m] if m.any() else h
    return x.mean(0), float(x.norm(dim=-1).median()), h[-1]


def spillover_coefs(D, w):
    """alpha = G^{-1} w for the Gram matrix G = D^T D (columns of D are unit directions)."""
    G = D.T @ D
    return torch.linalg.solve(G, w.to(D))


def train_spillover(bb, dtype):
    d = outdir(bb, "spillover"); os.makedirs(d, exist_ok=True)
    seed_all(SEED); t0 = time.time()
    tok, model = load(bb, dtype)
    layer = model.config.num_hidden_layers // 2
    diffs, norms = {o: [] for o in PATHWAYS}, []
    with torch.no_grad():
        for o, p in train_pairs():
            mc, nc, _ = _resid_resp_means(bb, tok, model, layer, p, p["chosen"])
            mr, nr, _ = _resid_resp_means(bb, tok, model, layer, p, p["rejected"])
            diffs[o].append(mc - mr); norms += [nc, nr]
    D = torch.stack([F.normalize(torch.stack(diffs[o]).mean(0), dim=0) for o in PATHWAYS], 1).double()  # d x N
    w = torch.full((len(PATHWAYS),), 1.0 / len(PATHWAYS), dtype=torch.double, device=D.device)
    coef = spillover_coefs(D, w)
    v = (D @ coef).float().cpu()
    err = float((D.T @ (D @ coef) - w).abs().max())
    hn = float(torch.tensor(norms).median())
    vp, res = val_pairs(), {}
    for ratio in (0.05, 0.1, 0.15, 0.2, 0.3):
        vec = v / v.norm() * ratio * hn
        h = blocks(model)[layer - 1].register_forward_hook(AddHook(vec.to(DEV, model.dtype)))
        res[ratio] = pref_accuracy(bb, tok, model, vp)
        h.remove()
    best = max(res, key=res.get)
    torch.save({"layer": layer, "vector": v / v.norm() * best * hn, "alpha": 1.0, "D": D.float().cpu(),
                "coef": coef.float().cpu(), "w": w.float().cpu()}, f"{d}/steer.pt")
    json.dump({"method": "spillover", "layer": layer, "val_acc": res, "ratio": best, "median_norm": hn,
               "gram": (D.T @ D).tolist(), "coef": coef.tolist(), "defining_property_err": err,
               "trainable_params": D.numel(), "train_sec": time.time() - t0, "seed": SEED},
              open(f"{d}/done.json", "w"), indent=1)


class MSRS(torch.nn.Module):
    def __init__(self, d, r, hidden=16):
        super().__init__()
        self.R = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(d, r, bias=False))
        self.W = torch.nn.Linear(d, r)
        self.mask = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, r))

    def forward(self, h):
        m = torch.sigmoid(self.mask(h))
        return h + ((self.W(h) - self.R(h)) * m) @ self.R.weight, m


class MSRSHook(ReFTHook):
    def __init__(self, mod, pos):
        super().__init__(lambda h: mod(h)[0], pos)


def _energy_rank(S, frac=0.6):
    c = torch.cumsum(S, 0) / S.sum()
    return int((c < frac).sum()) + 1


def msrs_subspaces(H, frac=0.6, cap=8):
    """H: list of (d x n_i) activation matrices per objective -> (S_align rows, sizes). Shared subspace from the SVD of
    the per-objective means, private subspaces from the residual SVDs; total rank capped at `cap`."""
    tau = torch.stack([h.mean(1) for h in H], 1)
    U, S, _ = torch.linalg.svd(tau, full_matrices=False)
    rs = min(_energy_rank(S, frac), cap - len(H))
    Bs = U[:, :rs].T
    priv, sizes = [], []
    for i, h in enumerate(H):
        res = h - Bs.T @ (Bs @ h)
        for B in priv:
            res = res - B.T @ (B @ res)
        Ui, Si, _ = torch.linalg.svd(res, full_matrices=False)
        priv.append(Ui[:, :_energy_rank(Si, frac)].T)
    budget = cap - rs
    want = [b.shape[0] for b in priv]
    while sum(want) > budget:  # shrink the largest private block first, keeping at least one direction each
        k = max(range(len(want)), key=lambda j: want[j])
        want[k] -= 1
    rows = torch.cat([Bs] + [b[:n] for b, n in zip(priv, want)])
    Q, _ = torch.linalg.qr(rows.T)
    return Q.T.contiguous(), [rs] + want


def train_msrs(bb, dtype, steps=3750, bs=4, lr=9e-4):
    steps = STEPS or steps
    d = outdir(bb, "msrs"); os.makedirs(d, exist_ok=True)
    seed_all(SEED); t0 = time.time()
    tok, model = load(bb, dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    L = model.config.num_hidden_layers
    layer = L // 2
    data = {o: read_jsonl(f"{DATA}/train_{o}.jsonl") for o in PATHWAYS}
    H = []
    with torch.no_grad():
        for o in PATHWAYS:
            rows = random.Random(SEED).sample(data[o], min(500, len(data[o])))
            cols = []
            for r in rows:
                ids, att, _ = batchify([encode(bb, tok, r, r["output"])], tok.pad_token_id)
                h = model(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states[layer][0].float()
                cols.append(h[-5:].mean(0))
            H.append(torch.stack(cols, 1).double().cpu())
    S_align, sizes = msrs_subspaces(H)
    r = S_align.shape[0]
    mod = MSRS(model.config.hidden_size, r).to(DEV)
    with torch.no_grad():
        mod.R.weight = S_align.float().to(DEV)
    Sa = S_align.float().to(DEV)
    pos = PromptPositions(first=0, last=5)
    blocks(model)[layer - 1].register_forward_hook(MSRSHook(mod, pos))
    store = {}
    mod.mask.register_forward_hook(lambda m_, i_, o_: store.__setitem__("m", torch.sigmoid(o_)))
    opt = torch.optim.Adam(mod.parameters(), lr=lr)
    ck = f"{d}/ckpt.pt"
    start = 0
    if os.path.exists(ck):
        c = torch.load(ck, map_location=DEV); mod.load_state_dict(c["mod"]); opt.load_state_dict(c["opt"]); start = c["step"]
    for step in range(start + 1, steps + 1):
        o = PATHWAYS[(step - 1) % 3]
        rng = random.Random(f"{SEED}-{step}")
        batch = [rng.choice(data[o]) for _ in range(bs)]
        ids, att, msk = batchify([encode(bb, tok, x, x["output"]) for x in batch], tok.pad_token_id)
        pos.mask = pos.select(att * (1 - msk.long()))
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            nll = F.cross_entropy(lg.transpose(1, 2), ids[:, 1:], reduction="none")
            task = (nll * msk[:, 1:]).sum() / msk[:, 1:].sum().clamp(min=1)
        m = store["m"].float()
        sel = pos.mask.to(m.device)
        reg = ((m[sel] - 1) ** 2).sum(-1).mean() if sel.any() else m.sum() * 0
        Rw = mod.R.weight
        align = 1 - (Rw * Sa).sum() / (Rw.norm() * Sa.norm())
        loss = task + 0.3 * reg + 0.5 * align
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        stop = should_stop() and step < steps
        if step % 250 == 0 or stop:
            torch.save({"mod": mod.state_dict(), "opt": opt.state_dict(), "step": step}, ck)
        if stop:
            stop_now(f"msrs step {step}")
    torch.save({"state": mod.state_dict(), "rank": r, "layer": layer, "sizes": sizes}, f"{d}/msrs.pt")
    json.dump({"method": "msrs", "layer": layer, "rank": r, "subspace_sizes": sizes, "steps": steps, "batch": bs,
               "lr": lr, "trainable_params": sum(p.numel() for p in mod.parameters()), "train_sec": time.time() - t0,
               "seed": SEED}, open(f"{d}/done.json", "w"), indent=1)


class KClassifier(torch.nn.Module):
    def __init__(self, d, n_cls, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden),
                                       torch.nn.ReLU(), torch.nn.Linear(hidden, n_cls))

    def forward(self, a):
        return self.net(a)


def ksteer_step(clf, a, alpha, pos, neg):
    """One K-Steering step on activations a (..., d): a - alpha * grad_a L, L = -mean(target logits) + mean(avoid logits),
    computed per position."""
    with torch.enable_grad():
        x = a.detach().float().requires_grad_(True)
        lg = clf(x)
        loss = (-lg[..., pos].mean(-1) + lg[..., neg].mean(-1)).sum()
        g, = torch.autograd.grad(loss, x)
    return (x - alpha * g).detach()


class KSteerHook:
    def __init__(self, clf, alpha, pos, neg):
        self.clf, self.alpha, self.pos, self.neg = clf, alpha, list(pos), list(neg)

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        new = ksteer_step(self.clf, hs, self.alpha, self.pos, self.neg).to(hs.dtype)
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new


def _coherence(bb, tok, model, prompts, layer, hook=None, max_new=64):
    """Greedy continuations (with an optional hook), scored by the unsteered model: (mean token NLL, mean length)."""
    from .common import build_prompt
    side = tok.padding_side
    tok.padding_side = "left"
    built = [build_prompt(bb, tok, p) for p in prompts]
    enc = tok([b[0] for b in built], return_tensors="pt", padding=True, add_special_tokens=built[0][1]).to(DEV)
    h = blocks(model)[layer - 1].register_forward_hook(hook) if hook is not None else None
    try:
        with torch.no_grad():
            g = model.generate(**enc, do_sample=False, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
    finally:
        if h is not None:
            h.remove()
        tok.padding_side = side
    P = enc.input_ids.shape[1]
    gen = g[:, P:]
    keep = torch.ones_like(gen)
    for i in range(gen.shape[0]):  # tokens after the first end-of-sequence do not count
        e = (gen[i] == tok.eos_token_id).nonzero()
        if len(e):
            keep[i, int(e[0]) + 1:] = 0
    att = torch.cat([enc.attention_mask, keep], 1)
    with torch.no_grad():
        lg = model(input_ids=g, attention_mask=att).logits[:, P - 1:-1].float()
    lp = torch.gather(lg.log_softmax(-1), 2, gen[..., None])[..., 0]
    m = keep.float()
    return float(-(lp * m).sum() / m.sum().clamp(min=1)), float(m.sum(1).mean())


def train_ksteer(bb, dtype, epochs=30, bs=32):
    d = outdir(bb, "ksteer"); os.makedirs(d, exist_ok=True)
    seed_all(SEED); t0 = time.time()
    tok, model = load(bb, dtype)
    layer = model.config.num_hidden_layers // 2
    X, Y = [], []
    with torch.no_grad():
        for o, p in train_pairs():
            t = PATHWAYS.index(o)
            for resp, c in ((p["chosen"], 2 * t), (p["rejected"], 2 * t + 1)):
                X.append(_resid_resp_means(bb, tok, model, layer, p, resp)[2].cpu()); Y.append(c)
    X, Y = torch.stack(X).to(DEV), torch.tensor(Y, device=DEV)
    n_cls = 2 * len(PATHWAYS)
    clf = KClassifier(X.shape[1], n_cls).to(DEV)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    for _ in range(epochs):
        perm = torch.randperm(len(Y), generator=g)
        for s in range(0, len(Y), bs):
            idx = perm[s:s + bs].to(DEV)
            loss = F.cross_entropy(clf(X[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = float((clf(X).argmax(-1) == Y).float().mean())
    pos, neg = [2 * t for t in range(len(PATHWAYS))], [2 * t + 1 for t in range(len(PATHWAYS))]
    vp, res, coh = val_pairs(), {}, {}
    prompts = [(p["instruction"] + ("\n" + p["input"] if p.get("input") else "")) for _, p in vp[::max(1, len(vp) // 16)]][:16]
    nll0, len0 = _coherence(bb, tok, model, prompts, layer)
    for a in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        nll, ln = _coherence(bb, tok, model, prompts, layer, KSteerHook(clf.eval(), a, pos, neg))
        coh[a] = {"nll": nll, "len": ln, "coherent": bool(nll - nll0 <= math.log(1.5) and ln >= 0.5 * len0)}
        if coh[a]["coherent"]:
            h = blocks(model)[layer - 1].register_forward_hook(KSteerHook(clf.eval(), a, pos, neg))
            res[a] = pref_accuracy(bb, tok, model, vp)
            h.remove()
    ok = []  # contiguous rule, stop at the first incoherent alpha in the ascending grid
    for a in coh:
        if not coh[a]["coherent"]:
            break
        ok.append(a)
    best = max(ok) if ok else min(coh)
    torch.save({"state": clf.state_dict(), "n_cls": n_cls, "layer": layer, "alpha": best, "pos": pos, "neg": neg},
               f"{d}/ksteer.pt")
    json.dump({"method": "ksteer", "selection": "largest_coherent_v48", "base_nll": nll0, "base_len": len0,
               "coherence": coh, "layer": layer, "train_acc": acc, "val_acc": res, "alpha": best,
               "trainable_params": sum(p.numel() for p in clf.parameters()), "train_sec": time.time() - t0,
               "seed": SEED}, open(f"{d}/done.json", "w"), indent=1)


TRAIN.update({"spillover": train_spillover, "msrs": train_msrs, "ksteer": train_ksteer})


def generate(bb, method, dtype="bfloat16", batch=32, seed=None, benches=None):
    seed = SEED if seed is None else seed
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .common import ALL_BACKBONES, MAX_NEW_TOKENS, build_prompt
    max_new = {"harmbench": 512}
    gd = f"{RUNS}/{bb}/gen/{method}-s{seed}__joint"
    os.makedirs(gd, exist_ok=True)
    repo = ALL_BACKBONES.get(bb, bb)
    tok = AutoTokenizer.from_pretrained(repo, padding_side="left", truncation_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=getattr(torch, dtype),
                                                 **({"device_map": DEV} if DEV == "cuda" else {})).to(DEV).eval()
    pos = attach(bb, model, tok, method)
    for bench in benches or BENCHES:
        path = f"{gd}/{bench}.jsonl"
        if os.path.exists(path):
            continue
        rows = read_jsonl(f"{DATA}/eval_{bench}.jsonl")
        part = f"{gd}/{bench}.partial.jsonl"
        done = {r["id"]: r["raw_output"] for r in read_jsonl(part)} if os.path.exists(part) else {}
        built = [build_prompt(bb, tok, r["instruction"]) for r in rows]
        texts, add_special = [b[0] for b in built], built[0][1]
        order = sorted([k for k, r in enumerate(rows) if r["id"] not in done], key=lambda k: len(texts[k]))
        outs = [done.get(r["id"]) for r in rows]
        with torch.no_grad():
            for bi, s in enumerate(range(0, len(order), batch)):
                seed_all(seed + 1000 * len(done) + bi)
                ks = order[s:s + batch]
                enc = tok([texts[k] for k in ks], return_tensors="pt", padding=True, truncation=True, max_length=1024,
                          add_special_tokens=add_special).to(DEV)
                if pos is not None:
                    pos.mask = pos.select(enc.attention_mask)
                g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.9, top_k=0,
                                   max_new_tokens=max_new.get(bench, MAX_NEW_TOKENS), pad_token_id=tok.pad_token_id)
                with open(part, "a") as f:
                    for j, k in enumerate(ks):
                        outs[k] = tok.decode(g[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
                        f.write(json.dumps({"id": rows[k]["id"], "raw_output": outs[k]}) + "\n")
                if should_stop() and s + batch < len(order):
                    stop_now(f"{method} {bench}")
        write_jsonl(path, [dict(id=r["id"], instruction=r["instruction"], output=clean_generation(o), raw_output=o)
                           for r, o in zip(rows, outs)])
        os.remove(part)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "generate"])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--method", required=True, choices=list(TRAIN))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--benches", default="")
    a = ap.parse_args()
    global SEED
    SEED = a.seed
    if a.cmd == "train":
        dj = f"{outdir(a.backbone, a.method)}/done.json"
        if os.path.exists(dj) and not (a.method == "ksteer" and json.load(open(dj)).get("selection") not in ("largest_coherent_v47", "largest_coherent_v48")):
            print("already trained"); return
        TRAIN[a.method](a.backbone, a.dtype)
    else:
        generate(a.backbone, a.method, a.dtype, benches=a.benches.split(",") if a.benches else None)


if __name__ == "__main__":
    main()
