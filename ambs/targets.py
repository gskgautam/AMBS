"""Supervision-target quality (Extended Analysis) and shared preference data for the matched baselines.

  python -m ambs.targets build                     # needs internet: stronger targets + preference pairs
  python -m ambs.targets selfdistill --backbone llama2-7b-chat   # GPU: the chat model's own greedy responses

Writes, for the same training prompts as the original data (data/train_{help,harm,honest}.jsonl):
  train_*.strong.jsonl        help: GPT-4-written Alpaca responses (vicgalle/alpaca-gpt4) for the same instructions
                              harm: PKU-SafeRLHF responses labelled safe and safer (8,142 sampled, seed 42; prompts in
                                    the BeaverTails test split removed)
                              honest: every correct answer of each training question (question-level split unchanged)
  train_*.self_<family>.jsonl the instruction-tuned model's greedy response to each original training prompt
  pairs_{help,harm,honest}.jsonl  (prompt, chosen, rejected) for activation steering and DPO:
                              help: GPT-4 response vs original Alpaca response; harm: safer vs less safe PKU-SafeRLHF
                              response; honest: correct vs incorrect TruthfulQA answer (training questions only)
"""
import argparse
import json
import os
import random
import time

from .common import ALL_BACKBONES, DATA, build_prompt, read_jsonl, write_jsonl
from .handoff import should_stop, stop_now

FAMILY = {"llama2-7b-chat": "llama2", "mistral-7b-instruct": "mistral"}
N_HARM = 8142


def build():
    from datasets import load_dataset
    rng = random.Random(42)
    help_rows = read_jsonl(f"{DATA}/train_help.jsonl")
    g4 = {(r["instruction"].strip(), (r["input"] or "").strip()): r["output"]
          for r in load_dataset("vicgalle/alpaca-gpt4", split="train")}
    strong_help, pairs_help = [], []
    for r in help_rows:
        out = g4.get((r["instruction"].strip(), (r.get("input") or "").strip()))
        if out:
            strong_help.append({**r, "output": out})
            pairs_help.append({"instruction": r["instruction"], "input": r.get("input", ""), "chosen": out,
                               "rejected": r["output"]})
    test_prompts = {r["instruction"].strip() for r in read_jsonl(f"{DATA}/eval_beavertails.jsonl")}
    pk = [r for r in load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train") if r["prompt"].strip() not in test_prompts]
    harm, pairs_harm = [], []
    for r in pk:
        s = r["safer_response_id"]
        if r[f"is_response_{s}_safe"]:
            harm.append((r["prompt"], r[f"response_{s}"], r[f"response_{1 - s}"]))
    rng.shuffle(harm)
    harm = harm[:N_HARM]
    strong_harm = [{"id": f"pk-{i}", "instruction": p, "input": "", "output": c, "pathway": "harm"}
                   for i, (p, c, _) in enumerate(harm)]
    pairs_harm = [{"instruction": p, "input": "", "chosen": c, "rejected": x} for p, c, x in harm]
    tqa = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    train_q = sorted({int(r["id"].split("-")[1]) for r in read_jsonl(f"{DATA}/train_honest.jsonl")})
    strong_hon, pairs_hon = [], []
    for qi in train_q:
        q = tqa[qi]
        for j, a in enumerate(q["correct_answers"]):
            strong_hon.append({"id": f"tqa-{qi}-{j}", "instruction": q["question"], "input": "", "output": a,
                               "pathway": "honest"})
        for c, x in zip(q["correct_answers"], q["incorrect_answers"]):
            pairs_hon.append({"instruction": q["question"], "input": "", "chosen": c, "rejected": x})
    for name, rows in (("train_help.strong", strong_help), ("train_harm.strong", strong_harm),
                       ("train_honest.strong", strong_hon), ("pairs_help", pairs_help), ("pairs_harm", pairs_harm),
                       ("pairs_honest", pairs_hon)):
        write_jsonl(f"{DATA}/{name}.jsonl", rows)
        print(f"{name}: {len(rows)}")


def selfdistill(bb, batch, dtype):
    """Greedy responses of an instruction-tuned model to every original training prompt (resumable per batch)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    fam = FAMILY[bb]
    part = f"{DATA}/selfdistill_{fam}.partial.jsonl"
    done = {r["id"]: r["output"] for r in read_jsonl(part)} if os.path.exists(part) else {}
    rows = [r for p in ("help", "harm", "honest") for r in read_jsonl(f"{DATA}/train_{p}.jsonl")]
    tok = AutoTokenizer.from_pretrained(ALL_BACKBONES[bb], padding_side="left", truncation_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(ALL_BACKBONES[bb], torch_dtype=getattr(torch, dtype),
                                                 device_map="cuda").eval()
    built = {r["id"]: build_prompt(bb, tok, r["instruction"], r.get("input", "")) for r in rows}
    todo = sorted([r for r in rows if r["id"] not in done], key=lambda r: len(built[r["id"]][0]))
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(todo), batch):
            chunk = todo[s:s + batch]
            enc = tok([built[r["id"]][0] for r in chunk], return_tensors="pt", padding=True, truncation=True,
                      max_length=768, add_special_tokens=built[chunk[0]["id"]][1]).to("cuda")
            out = model.generate(**enc, do_sample=False, max_new_tokens=256, pad_token_id=tok.pad_token_id)
            with open(part, "a") as f:
                for j, r in enumerate(chunk):
                    f.write(json.dumps({"id": r["id"], "output": tok.decode(out[j, enc.input_ids.shape[1]:],
                                                                              skip_special_tokens=True).strip()}) + "\n")
            print(f"selfdistill {bb} {len(done) + s + len(chunk)}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
            if should_stop() and s + batch < len(todo):
                stop_now(f"self-distillation sample {len(done) + s + len(chunk)}")
    done = {r["id"]: r["output"] for r in read_jsonl(part)}
    for p in ("help", "harm", "honest"):
        out = [{**r, "output": done[r["id"]]} for r in read_jsonl(f"{DATA}/train_{p}.jsonl") if done.get(r["id"])]
        write_jsonl(f"{DATA}/train_{p}.self_{fam}.jsonl", out)
        print(f"train_{p}.self_{fam}: {len(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "selfdistill"])
    ap.add_argument("--backbone", default="llama2-7b-chat")
    ap.add_argument("--batch", type=int, default=int(os.environ.get("AMBS_GEN_BATCH", "64")))
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    build() if a.cmd == "build" else selfdistill(a.backbone, a.batch, a.dtype)


if __name__ == "__main__":
    main()
