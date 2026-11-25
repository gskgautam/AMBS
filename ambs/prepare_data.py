"""Build train/eval splits from the raw HF datasets (fixed defaults).

Outputs (in $AMBS_ROOT/data):
  train_help.jsonl     Alpaca, 20k sampled with seed 42, AlpacaEval-overlapping instructions removed
  train_harm.jsonl     BeaverTails 30k_train, is_safe=True, prompts overlapping 30k_test removed
  train_honest.jsonl   TruthfulQA train half; one row per (question, best_answer/correct_answer)
  eval_alpacaeval.jsonl  805 AlpacaEval instructions + text-davinci-003 reference outputs
  eval_truthfulqa.jsonl  held-out TruthfulQA questions + correct/incorrect answers
  eval_beavertails.jsonl BeaverTails 30k_test (all rows)
  data_stats.json
"""
import json
import os
import random

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from .common import DATA, write_json, write_jsonl

SEED = 42


def norm(s):
    return " ".join((s or "").lower().split())


def main():
    os.makedirs(DATA, exist_ok=True)
    stats = {}

    # ---- AlpacaEval (evaluation, help) ----
    ae_path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    ae = json.load(open(ae_path))
    gens = sorted({r.get("generator") for r in ae})
    eval_ae = [
        {"id": f"ae-{i}", "instruction": r["instruction"], "reference": r["output"], "dataset": r.get("dataset")}
        for i, r in enumerate(ae)
    ]
    write_jsonl(f"{DATA}/eval_alpacaeval.jsonl", eval_ae)
    stats["alpacaeval"] = {"n": len(eval_ae), "reference_generator": gens}
    ae_instr = {norm(r["instruction"]) for r in ae}

    # ---- Alpaca (train, help) ----
    alp = load_dataset("tatsu-lab/alpaca", split="train")
    rows = [
        {"instruction": r["instruction"], "input": r["input"], "output": r["output"]}
        for r in alp
        if norm(r["instruction"]) not in ae_instr and r["output"].strip()
    ]
    n_overlap = len(alp) - len(rows)
    random.Random(SEED).shuffle(rows)
    help_rows = [dict(r, id=f"alp-{i}", pathway="help") for i, r in enumerate(rows[:20000])]
    write_jsonl(f"{DATA}/train_help.jsonl", help_rows)
    stats["alpaca_train"] = {"source_n": len(alp), "dropped_overlap_or_empty": n_overlap, "n": len(help_rows)}

    # ---- BeaverTails ----
    bt_test = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    bt_train = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    eval_bt = [
        {"id": f"bt-{i}", "instruction": r["prompt"], "category": {k: v for k, v in r["category"].items() if v}}
        for i, r in enumerate(bt_test)
    ]
    write_jsonl(f"{DATA}/eval_beavertails.jsonl", eval_bt)
    test_prompts = {norm(r["prompt"]) for r in bt_test}
    bt_train_kept = [r for r in bt_train if norm(r["prompt"]) not in test_prompts and r["response"].strip()]
    safe = [r for r in bt_train if r["is_safe"]]
    harm_rows = [
        {"id": f"bt-train-{i}", "instruction": r["prompt"], "input": "", "output": r["response"], "pathway": "harm"}
        for i, r in enumerate(safe)
        if norm(r["prompt"]) not in test_prompts and r["response"].strip()
    ]
    write_jsonl(f"{DATA}/train_harm.jsonl", harm_rows)
    stats["beavertails"] = {
        "test_n": len(eval_bt),
        "test_unique_prompts": len(test_prompts),
        "train_source_n": len(bt_train),
        "train_is_safe_n": len(safe),
        "train_after_overlap_drop_n": len(harm_rows),
    }

    # ---- TruthfulQA (generation config) ----
    tqa = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    qidx = list(range(len(tqa)))
    random.Random(SEED).shuffle(qidx)
    half = len(qidx) // 2
    train_q, test_q = sorted(qidx[:half]), sorted(qidx[half:])
    honest_rows = []
    for qi in train_q:
        r = tqa[qi]
        # Paper (Method): each question is paired with its original best_answer as the Honesty target.
        honest_rows.append({"id": f"tqa-{qi}-0", "instruction": r["question"], "input": "",
                            "output": r["best_answer"], "pathway": "honest"})
    write_jsonl(f"{DATA}/train_honest.jsonl", honest_rows)
    eval_tqa = [
        {
            "id": f"tqa-{qi}",
            "instruction": tqa[qi]["question"],
            "best_answer": tqa[qi]["best_answer"],
            "correct_answers": tqa[qi]["correct_answers"],
            "incorrect_answers": tqa[qi]["incorrect_answers"],
            "category": tqa[qi]["category"],
        }
        for qi in test_q
    ]
    write_jsonl(f"{DATA}/eval_truthfulqa.jsonl", eval_tqa)
    stats["truthfulqa"] = {
        "questions": len(tqa),
        "train_questions": len(train_q),
        "test_questions": len(test_q),
        "train_rows": len(honest_rows),
    }

    # ---- Table 3 label perturbation: flip each binary label w.p. 0.05 (fixed seed) ----
    # BeaverTails: is_safe flipped, then the usual is_safe=True filter. TruthfulQA: the target's truthful label is
    # flipped, i.e. best_answer replaced by an incorrect answer. Alpaca has no binary labels (unchanged).
    rng = random.Random(SEED + 1)
    flips = 0
    harm_p = []
    for i, r in enumerate(bt_train_kept):
        lab = bool(r["is_safe"])
        if rng.random() < 0.05:
            lab, flips = not lab, flips + 1
        if lab:
            harm_p.append({"id": f"bt-train-p{i}", "instruction": r["prompt"], "input": "", "output": r["response"],
                           "pathway": "harm"})
    hon_p, tflips = [], 0
    for r in honest_rows:
        qi = int(r["id"].split("-")[1])
        out = r["output"]
        if rng.random() < 0.05 and tqa[qi]["incorrect_answers"]:
            out, tflips = tqa[qi]["incorrect_answers"][0], tflips + 1
        hon_p.append(dict(r, output=out))
    write_jsonl(f"{DATA}/train_help.pert.jsonl", help_rows)
    write_jsonl(f"{DATA}/train_harm.pert.jsonl", harm_p)
    write_jsonl(f"{DATA}/train_honest.pert.jsonl", hon_p)
    stats["perturbation"] = {"beavertails_rows": len(bt_train_kept), "beavertails_flips": flips,
                             "harm_rows": len(harm_p), "truthfulqa_flips": tflips, "honest_rows": len(hon_p)}

    # ---- HarmBench (Table 5): DirectRequest on standard + contextual test behaviours ----
    hb = os.path.join(DATA, "raw", "harmbench_behaviors_text_test.csv")
    if os.path.exists(hb):
        import csv
        rows = [r for r in csv.DictReader(open(hb)) if r["FunctionalCategory"] in ("standard", "contextual")]
        ev = []
        for r in rows:
            tc = r["Behavior"] if not r["ContextString"] else f"{r['ContextString']}\n\n---\n\n{r['Behavior']}"
            ev.append({"id": r["BehaviorID"], "instruction": tc, "behavior": r["Behavior"],
                       "context": r["ContextString"], "category": r["FunctionalCategory"]})
        write_jsonl(f"{DATA}/eval_harmbench.jsonl", ev)
        stats["harmbench"] = {"n": len(ev), "standard": sum(e["category"] == "standard" for e in ev),
                              "contextual": sum(e["category"] == "contextual" for e in ev)}

    write_json(f"{DATA}/data_stats.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
