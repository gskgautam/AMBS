"""Cache the frozen backbone's final (post-norm) hidden states at every response-token-predicting position.

Because AMBS only adds offsets after the final norm and the backbone is frozen, these hidden states never
change during training; caching them once gives exactly the same gradients as recomputing them each step.

Writes, per pathway, in runs/<backbone>/cache/:
  <pathway>.h.npy    float16 (Ntok, d)  hidden state at position t (predicts token t+1)
  <pathway>.y.npy    int32   (Ntok,)    target token id (token t+1)
  <pathway>.off.npy  int64   (Nex+1,)   example boundaries into the token arrays
  meta.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import ALL_BACKBONES as BACKBONES, DATA, MAX_LEN, PATHWAYS, RUNS, build_prompt, read_jsonl, write_json


def tokenize_example(tok, r, bb="llama2-7b"):
    prompt, add_special, resp_prefix = build_prompt(bb, tok, r["instruction"], r.get("input", ""))
    p = tok(prompt, add_special_tokens=add_special).input_ids
    # Tokenize prompt+response together so the response's first token is the one the model would generate
    # after the prompt (separate tokenization adds a spurious word-start marker for Llama/Mistral).
    full = tok(prompt + resp_prefix + r["output"], add_special_tokens=add_special).input_ids
    if full[: len(p)] == p and len(full) > len(p):
        o = full[len(p):]
    else:
        o = tok(r["output"], add_special_tokens=False).input_ids
    if o and o[-1] == tok.eos_token_id:
        o = o[:-1]
    ids = (p + o + [tok.eos_token_id])[:MAX_LEN]
    n_resp = len(ids) - len(p)
    return ids, len(p), n_resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--data_tag", default="", help="'pert' = Table 3 label-perturbed training data")
    a = ap.parse_args()
    repo = BACKBONES[a.backbone]
    out = f"{RUNS}/{a.backbone}/cache" + (f"-{a.data_tag}" if a.data_tag else "")
    suffix = f".{a.data_tag}" if a.data_tag else ""
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/meta.json"):
        print("cache exists, skipping")
        return

    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=getattr(torch, a.dtype), device_map="cuda")
    model.eval()
    d = model.config.hidden_size
    meta = {"data_tag": a.data_tag, "dtype": a.dtype, "backbone": repo, "d": d, "vocab": model.config.vocab_size, "pathways": {}}

    for pw in PATHWAYS:
        t0 = time.time()
        rows = read_jsonl(f"{DATA}/train_{pw}{suffix}.jsonl")
        toks = []
        skipped = 0
        for r in rows:
            ids, plen, n_resp = tokenize_example(tok, r, a.backbone)
            if n_resp <= 0:
                skipped += 1
                continue
            toks.append((r["id"], ids, plen, n_resp))
        n_tok = sum(t[3] for t in toks)
        H = np.lib.format.open_memmap(f"{out}/{pw}.h.npy.{os.getpid()}.tmp", mode="w+", dtype=np.float16, shape=(n_tok, d))
        Y = np.zeros(n_tok, dtype=np.int32)
        off = np.zeros(len(toks) + 1, dtype=np.int64)
        for k, t in enumerate(toks):
            off[k + 1] = off[k] + t[3]
        order = sorted(range(len(toks)), key=lambda k: len(toks[k][1]))
        with torch.no_grad():
            for s in range(0, len(order), a.batch):
                ks = order[s : s + a.batch]
                L = max(len(toks[k][1]) for k in ks)
                ids = torch.full((len(ks), L), tok.pad_token_id or 0, dtype=torch.long)
                att = torch.zeros((len(ks), L), dtype=torch.long)
                for j, k in enumerate(ks):
                    x = toks[k][1]
                    ids[j, : len(x)] = torch.tensor(x)
                    att[j, : len(x)] = 1
                hs = model.model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
                if not torch.isfinite(hs[att.cuda().bool()]).all():
                    raise RuntimeError(f"non-finite hidden states in {a.dtype}; rerun with --dtype bfloat16")
                for j, k in enumerate(ks):
                    _, x, plen, n_resp = toks[k]
                    # positions plen-1 .. len(x)-2 predict tokens plen .. len(x)-1
                    H[off[k] : off[k + 1]] = hs[j, plen - 1 : len(x) - 1].float().cpu().numpy().astype(np.float16)
                    Y[off[k] : off[k + 1]] = np.array(x[plen:], dtype=np.int32)
                if (s // a.batch) % 100 == 0:
                    print(f"{pw}: {s}/{len(order)} ({time.time()-t0:.0f}s)", flush=True)
        H.flush()
        del H
        os.replace(f"{out}/{pw}.h.npy.{os.getpid()}.tmp", f"{out}/{pw}.h.npy")
        np.save(f"{out}/{pw}.y.npy", Y)
        np.save(f"{out}/{pw}.off.npy", off)
        meta["pathways"][pw] = {
            "examples": len(toks),
            "skipped_no_response_tokens": skipped,
            "tokens": int(n_tok),
            "ids": [t[0] for t in toks][:5] + ["..."],
            "seconds": round(time.time() - t0),
        }
        print(json.dumps({pw: meta["pathways"][pw]}), flush=True)
    write_json(f"{out}/meta.json", meta)


if __name__ == "__main__":
    main()
