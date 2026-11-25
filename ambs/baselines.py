"""Matched baselines under the AMBS protocol (Extended Analysis): same backbone, training prompts, decoding and judges.

  python -m ambs.baselines --method caa      --backbone llama2-7b --out runs/llama2-7b/caa-s42
  python -m ambs.baselines --method lora_sft --backbone llama2-7b --out runs/llama2-7b/lora_sft-s42
  python -m ambs.baselines --method dpo      --backbone llama2-7b --out runs/llama2-7b/dpo-s42

caa       Contrastive activation addition: for each objective, the mean difference of residual-stream activations
          (block L/2, averaged over response tokens) between chosen and rejected responses (data/pairs_*.jsonl, up to
          512 pairs per objective); the three vectors are averaged with the same equal weights as AMBS pooling and
          added with multiplier 1 at every position during generation.
lora_sft  LoRA (r=8, alpha=16, dropout 0.05, q/v projections) trained with next-token loss on the original AMBS
          training targets, objectives sampled in turn with batch 16, 1,250 steps, lr 2e-4.
dpo       LoRA DPO (beta 0.1) on the preference pairs, objectives sampled in turn with batch 8 pairs, 1,000 steps,
          lr 5e-5; the reference model is the backbone with the adapter disabled.
Training checkpoints every 100 steps and yields to an H200 twin at a checkpoint (ambs/handoff.py); batches are drawn
from a per-step random stream, so a resumed run trains on the same batches (dropout noise is not replayed).
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import ALL_BACKBONES, DATA, build_prompt, read_jsonl, seed_all
from .handoff import should_stop, stop_now

OBJ = ("help", "harm", "honest")
DEV = os.environ.get("AMBS_DEVICE", "cuda")  # tests run on CPU
STEPS = {"lora_sft": int(os.environ.get("AMBS_BASELINE_STEPS", 1250)), "dpo": int(os.environ.get("AMBS_BASELINE_STEPS", 1000))}


def load(bb, dtype):
    tok = AutoTokenizer.from_pretrained(ALL_BACKBONES.get(bb, bb))
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(ALL_BACKBONES.get(bb, bb), torch_dtype=getattr(torch, dtype), **({"device_map": DEV} if DEV == "cuda" else {})).to(DEV)
    return tok, model


def encode(bb, tok, r, resp, max_len=512):
    """Prompt + response token ids with a mask over response tokens (same joint tokenization as ambs/cache.py)."""
    prompt, add_special, prefix = build_prompt(bb, tok, r["instruction"], r.get("input", ""))
    p_ids = tok(prompt, add_special_tokens=add_special).input_ids
    full = tok(prompt + prefix + resp, add_special_tokens=add_special).input_ids + [tok.eos_token_id]
    full = full[:max_len]
    mask = [0] * min(len(p_ids), len(full)) + [1] * max(0, len(full) - len(p_ids))
    return full, mask


def batchify(seqs, pad):
    T = max(len(s) for s, _ in seqs)
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    msk = torch.zeros((len(seqs), T), dtype=torch.float)
    for i, (s, m) in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1; msk[i, :len(m)] = torch.tensor(m, dtype=torch.float)
    return ids.to(DEV), att.to(DEV), msk.to(DEV)


def caa(bb, out, dtype, n=512, seed=42):
    tok, model = load(bb, dtype)
    L = model.config.num_hidden_layers
    layer = L // 2
    vecs = []
    with torch.no_grad():
        for o in OBJ:
            pairs = read_jsonl(f"{DATA}/pairs_{o}.jsonl")
            random.Random(seed).shuffle(pairs)
            acc = None
            for p in pairs[:n]:
                diff = []
                for resp in (p["chosen"], p["rejected"]):
                    ids, att, msk = batchify([encode(bb, tok, p, resp)], tok.pad_token_id)
                    h = model(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states[layer][0].float()
                    m = msk[0].bool()
                    diff.append(h[m].mean(0) if m.any() else h.mean(0))
                d = diff[0] - diff[1]
                acc = d if acc is None else acc + d
            vecs.append(acc / min(n, len(pairs)))
    os.makedirs(out, exist_ok=True)
    torch.save({"layer": layer, "vector": torch.stack(vecs).mean(0).cpu(), "alpha": 1.0}, f"{out}/steer.pt")
    json.dump({"method": "caa", "layer": layer, "pairs_per_objective": n}, open(f"{out}/done.json", "w"))


def train_lora(bb, out, dtype, method, seed=42):
    from peft import LoraConfig, PeftModel, get_peft_model
    seed_all(seed)
    tok, model = load(bb, dtype)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    os.makedirs(out, exist_ok=True)
    ck = f"{out}/ckpt"
    if os.path.exists(f"{ck}/adapter_config.json"):
        model = PeftModel.from_pretrained(model, ck, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                                 target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    lr, steps, bs = (2e-4, STEPS[method], 16) if method == "lora_sft" else (5e-5, STEPS[method], 8)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    state = json.load(open(f"{ck}/state.json")) if os.path.exists(f"{ck}/state.json") else {"step": 0, "sec": 0.0}
    if os.path.exists(f"{ck}/opt.pt"):
        opt.load_state_dict(torch.load(f"{ck}/opt.pt"))
    if method == "lora_sft":
        data = {o: read_jsonl(f"{DATA}/train_{o}.jsonl") for o in OBJ}
    else:
        data = {o: read_jsonl(f"{DATA}/pairs_{o}.jsonl") for o in OBJ}
    t0 = time.time()
    log = open(f"{out}/train_log.jsonl", "a")
    for step in range(state["step"] + 1, steps + 1):
        o = OBJ[(step - 1) % 3]
        rng = random.Random(f"{seed}-{step}")  # per-step stream: a resumed run sees exactly the batches of an uninterrupted one
        batch = [rng.choice(data[o]) for _ in range(bs)]
        with torch.autocast(DEV, dtype=torch.bfloat16):
            if method == "lora_sft":
                ids, att, msk = batchify([encode(bb, tok, r, r["output"]) for r in batch], tok.pad_token_id)
                logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
                nll = F.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")
                loss = (nll * msk[:, 1:]).sum() / msk[:, 1:].sum().clamp(min=1)
            else:
                seqs = [encode(bb, tok, r, r["chosen"]) for r in batch] + [encode(bb, tok, r, r["rejected"]) for r in batch]
                ids, att, msk = batchify(seqs, tok.pad_token_id)

                def seq_logp(m):
                    lg = m(input_ids=ids, attention_mask=att).logits[:, :-1].float()
                    lp = torch.gather(lg.log_softmax(-1), 2, ids[:, 1:, None])[..., 0]
                    return (lp * msk[:, 1:]).sum(1)
                pol = seq_logp(model)
                with torch.no_grad(), model.disable_adapter():
                    ref = seq_logp(model)
                margin = (pol[:bs] - ref[:bs]) - (pol[bs:] - ref[bs:])
                loss = -F.logsigmoid(0.1 * margin).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 25 == 0:
            log.write(json.dumps({"step": step, "objective": o, "loss": loss.item(),
                                  "sec": state["sec"] + time.time() - t0}) + "\n"); log.flush()
        stop = should_stop() and step < steps
        if step % 100 == 0 or stop or step == steps:
            model.save_pretrained(ck)
            torch.save(opt.state_dict(), f"{ck}/opt.pt")
            json.dump({"step": step, "sec": state["sec"] + time.time() - t0}, open(f"{ck}/state.json", "w"))
        if stop:
            stop_now(f"{method} step {step}")
    model.save_pretrained(out)
    n = sum(p.numel() for p in params)
    json.dump({"method": method, "steps": steps, "batch": bs, "lr": lr, "trainable_params": n,
               "train_sec": state["sec"] + time.time() - t0}, open(f"{out}/done.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["caa", "lora_sft", "dpo"])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if os.path.exists(f"{a.out}/done.json"):
        print("already trained:", a.out); return
    caa(a.backbone, a.out, a.dtype, seed=a.seed) if a.method == "caa" else train_lora(a.backbone, a.out, a.dtype, a.method, seed=a.seed)


if __name__ == "__main__":
    main()
