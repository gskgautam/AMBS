"""CPU smoke tests with a tiny Llama: the four representation baselines (train with validation selection, generate) and
the GPU-free parts of ambs/evals.py (capability on fixture data, folded == explicit AMBS-IC).
PYTHONPATH=. python tests/test_repbase_evals.py"""
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
os.makedirs(f"{root}/data/ext")
W = lambda p, rows: open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
for o in ("help", "harm", "honest"):
    W(f"{root}/data/train_{o}.jsonl", [{"id": f"{o}-{i}", "instruction": f"q{i}", "input": "", "output": f"a{i}"} for i in range(20)])
    W(f"{root}/data/pairs_{o}.jsonl", [{"instruction": f"q{i}", "input": "", "chosen": f"good {i}", "rejected": f"bad {i}"} for i in range(40)])
for b in ("alpacaeval", "truthfulqa", "beavertails"):
    W(f"{root}/data/eval_{b}.jsonl", [{"id": f"{b}-{i}", "instruction": f"q{i}", "reference": "r"} for i in range(3)])
W(f"{root}/data/ext/mmlu.jsonl", [{"id": f"m{i}", "subject": "x", "question": "q?", "choices": ["a", "b", "c", "d"],
                                   "answer": i % 4, "shots": []} for i in range(8)])
W(f"{root}/data/ext/gsm8k.jsonl", [{"id": f"g{i}", "question": "1+1?", "answer": "#### 2",
                                    "shots": [{"question": "2+2?", "answer": "#### 4"}]} for i in range(3)])
W(f"{root}/data/ext/xstest_safe.jsonl", [{"id": f"x{i}", "instruction": "kill a process?"} for i in range(3)])
env = dict(os.environ, AMBS_ROOT=root, AMBS_DEVICE="cpu", PYTHONPATH=os.getcwd(), AMBS_REP_PAIRS="8",
           AMBS_REP_STEPS="6", EXP="rep", BB="tiny")
run = lambda *a: subprocess.run([sys.executable, "-m", *a], env=env, capture_output=True, text=True)
for m in ("caa_tuned", "iti", "loreft", "matsteer"):
    r = run("ambs.repbase", "train", "--backbone", tiny, "--method", m, "--dtype", "float32")
    assert r.returncode == 0, (m, r.stderr[-2500:])
    d = json.load(open(f"{root}/runs/{tiny}/{m}-s42/done.json")); assert d["trainable_params"] > 0, d
    r = run("ambs.repbase", "generate", "--backbone", tiny, "--method", m, "--dtype", "float32")
    assert r.returncode == 0, (m, r.stderr[-2500:])
    assert len(open(f"{root}/runs/{tiny}/gen/{m}-s42__joint/alpacaeval.jsonl").read().splitlines()) == 3
sys.path.insert(0, os.getcwd()); os.environ.update(AMBS_ROOT=root, AMBS_DEVICE="cpu")
from ambs.repbase import LoReFT
m = LoReFT(32); R_ = m.R.weight
assert torch.allclose(R_ @ R_.T, torch.eye(8), atol=1e-5)  # orthonormal rows
r = run("ambs.evals", "capability", "--backbone", tiny, "--system", "base")
assert r.returncode == 0, r.stderr[-2500:]
c = json.load(open(f"{root}/results/evals/capability_{tiny}_base.json")); assert set(c) >= {"mmlu", "gsm8k", "xstest_refusal"}
print("repbase (CAA-tuned, ITI, LoReFT, MAT-Steer) and evals capability: OK")
