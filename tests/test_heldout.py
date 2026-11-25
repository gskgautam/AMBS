"""Held-out loss evaluator on a tiny random model (CPU): base vs a trained layer run, pooled decoder through the hook.
PYTHONPATH=. python tests/test_heldout.py"""
import json, os, shutil, subprocess, sys, tempfile
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers

root = tempfile.mkdtemp(); tiny = f"{root}/tiny"
cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=4, bos_token_id=1, eos_token_id=2)
torch.manual_seed(0); LlamaForCausalLM(cfg).save_pretrained(tiny)
vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, **{chr(i): i - 29 for i in range(32, 127)}}
tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Split("", "isolated")
PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>").save_pretrained(tiny)
os.makedirs(f"{root}/data")
for pw, n in [("help", 10), ("harm", 6), ("honest", 3)]:
    rows = [json.dumps({"id": f"{pw}{i}", "instruction": f"q{i}", "input": "", "output": f"{pw} answer"}) for i in range(n)]
    for sfx in ("", ".val", ".heldout"):
        open(f"{root}/data/train_{pw}{sfx}.jsonl", "w").write("\n".join(rows) + "\n")
env = dict(os.environ, AMBS_ROOT=root, AMBS_DEVICE="cpu")
r = subprocess.run([sys.executable, "-m", "ambs.train_layer", "--backbone", tiny, "--layer", "2", "--batch", "2",
                    "--epochs", "30", "--lr", "0.05"], env=env, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-2000:]
run = "constant-r8-lam0.1-lr0.05-b2-layer2-s42"
for split in ("val", "heldout"):
    r = subprocess.run([sys.executable, "-m", "ambs.heldout", "eval", "--backbone", tiny, "--split", split, "--runs", f"base,{run}"],
                       env=env, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-2000:]
    res = json.load(open(f"{root}/runs/{tiny}/loss_{split}.json"))
    assert res[run]["mix"] < res["base"]["mix"], res
    assert abs(res[run]["mix"] - sum(res[run][p] for p in ("help", "harm", "honest"))) < 1e-6
    r = subprocess.run([sys.executable, "-m", "ambs.heldout", "eval", "--backbone", tiny, "--split", split, "--runs", f"base,{run}"],
                       env=env, capture_output=True, text=True); assert "nothing to do" in r.stdout
print("held-out / validation loss evaluator: trained run beats base, resumable: OK")
