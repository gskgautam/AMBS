"""Matched baselines on CPU with a tiny Llama: CAA vector, LoRA-SFT and DPO train (with checkpoint + handoff stop and
resume), and generation with the adapter / steering vector.  PYTHONPATH=. python tests/test_baselines.py"""
import json, os, subprocess, sys, tempfile
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
root = tempfile.mkdtemp(); tiny = f"{root}/tiny"
cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=4, bos_token_id=1, eos_token_id=2)
torch.manual_seed(0); LlamaForCausalLM(cfg).save_pretrained(tiny)
vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, **{chr(i): i - 29 for i in range(32, 127)}}
tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Split("", "isolated")
PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                        model_input_names=["input_ids", "attention_mask"]).save_pretrained(tiny)
os.makedirs(f"{root}/data")
W = lambda p, rows: open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
for o in ("help", "harm", "honest"):
    W(f"{root}/data/train_{o}.jsonl", [{"id": f"{o}-{i}", "instruction": f"q{i}", "input": "", "output": f"a{i}"} for i in range(20)])
    W(f"{root}/data/pairs_{o}.jsonl", [{"instruction": f"q{i}", "input": "", "chosen": f"good {i}", "rejected": f"bad {i}"} for i in range(20)])
W(f"{root}/data/eval_alpacaeval.jsonl", [{"id": f"ae-{i}", "instruction": f"q{i}", "reference": "r"} for i in range(4)])
env = dict(os.environ, AMBS_ROOT=root, AMBS_DEVICE="cpu", PYTHONPATH=os.getcwd(), AMBS_BASELINE_STEPS="150", EXP="baselines", BB="tiny")
def run(*a, extra=None):
    return subprocess.run([sys.executable, "-m", *a], env={**env, **(extra or {})}, capture_output=True, text=True)
r = run("ambs.baselines", "--method", "caa", "--backbone", tiny, "--out", f"{root}/caa", "--dtype", "float32")
assert r.returncode == 0, r.stderr[-1500:]; st = torch.load(f"{root}/caa/steer.pt"); assert st["vector"].shape == (32,) and st["layer"] == 2
for m in ("lora_sft", "dpo"):
    out = f"{root}/{m}"; L = f"{root}/locks/baselines-tiny"; os.makedirs(L, exist_ok=True)
    open(f"{L}/takeover", "w").write("x")       # A100 asked to stop at the first checkpoint opportunity
    r = run("ambs.baselines", "--method", m, "--backbone", tiny, "--out", out, "--dtype", "float32", extra={"AMBS_ROLE": "a100"})
    assert r.returncode == 99 and os.path.exists(f"{out}/ckpt/state.json"), (m, r.returncode, r.stderr[-1500:])
    os.remove(f"{L}/takeover")
    r = run("ambs.baselines", "--method", m, "--backbone", tiny, "--out", out, "--dtype", "float32")
    assert r.returncode == 0, (m, r.stderr[-1500:])
    d = json.load(open(f"{out}/done.json")); assert d["steps"] == 150 and d["trainable_params"] > 0
    assert os.path.exists(f"{out}/adapter_config.json")
for tag, extra in (("caa-s42__joint", ["--steer", f"{root}/caa/steer.pt"]), ("lora_sft-s42__joint", ["--adapter", f"{root}/lora_sft"])):
    r = run("ambs.generate", "--backbone", tiny, "--mode", "base", "--tag", tag, "--benches", "alpacaeval", "--batch", "2", "--dtype", "float32", *extra)
    assert r.returncode == 0, (tag, r.stderr[-1500:])
    assert len(open(f"{root}/runs/{tiny}/gen/{tag}/alpacaeval.jsonl").read().splitlines()) == 4
print("baselines: CAA, LoRA-SFT and DPO (stop/resume), adapter and steering generation: OK")
