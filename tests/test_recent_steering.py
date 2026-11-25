"""CPU tests for the recent steering baselines: defining properties, then train + generate with a tiny Llama.
PYTHONPATH=. AMBS_DEVICE=cpu python tests/test_recent_steering.py"""
import json, os, subprocess, sys, tempfile
import torch

os.environ.setdefault("AMBS_DEVICE", "cpu")
sys.path.insert(0, os.getcwd())
from ambs.repbase import KClassifier, MSRS, ksteer_step, msrs_subspaces, spillover_coefs

# spillover: d_j^T (D alpha) = w_j for correlated unit directions
torch.manual_seed(0)
D = torch.nn.functional.normalize(torch.randn(64, 3, dtype=torch.double) + 2.0, dim=0)
w = torch.full((3,), 1 / 3, dtype=torch.double)
a = spillover_coefs(D, w)
assert torch.allclose(D.T @ (D @ a), w, atol=1e-9), D.T @ (D @ a)
naive = D.T @ (D @ w)
assert (naive - w).abs().max() > 0.05  # without the correction the directions spill over

# msrs: orthonormal rows, rank capped at 8, R initialised to S_align
H = [torch.randn(32, 40, dtype=torch.double) + 3 * torch.randn(32, 1, dtype=torch.double) for _ in range(3)]
S, sizes = msrs_subspaces(H)
assert S.shape[0] == sum(sizes) <= 8 and all(z >= 1 for z in sizes[1:]), sizes
assert torch.allclose(S @ S.T, torch.eye(S.shape[0], dtype=torch.double), atol=1e-8)
m = MSRS(32, S.shape[0])
with torch.no_grad():
    m.R.weight = S.float()
assert torch.allclose(m.R.weight, S.float(), atol=1e-5)
h = torch.randn(4, 32)
out, gate = m(h)
assert out.shape == h.shape and gate.min() >= 0 and gate.max() <= 1

# ksteer: a step raises the target logits and lowers the avoid logits
clf = KClassifier(32, 6)
x = torch.randn(5, 32)
pos, neg = [0, 2, 4], [1, 3, 5]
y = ksteer_step(clf, x, 0.05, pos, neg)
s = lambda z: (clf(z)[:, pos].mean(-1) - clf(z)[:, neg].mean(-1)).mean()
with torch.no_grad():
    assert s(y) > s(x), (s(y), s(x))
print("properties ok")

# end to end on a tiny model (reuses the fixture of tests/test_repbase_evals.py)
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
    W(f"{root}/data/train_{o}.jsonl", [{"id": f"{o}-{i}", "instruction": f"q{i}", "input": "", "output": f"a{i} b"} for i in range(20)])
    W(f"{root}/data/pairs_{o}.jsonl", [{"instruction": f"q{i}", "input": "", "chosen": f"good {i}", "rejected": f"bad {i}"} for i in range(40)])
for b in ("alpacaeval", "truthfulqa", "beavertails", "harmbench"):
    W(f"{root}/data/eval_{b}.jsonl", [{"id": f"{b}-{i}", "instruction": f"q{i}", "reference": "r"} for i in range(3)])
env = dict(os.environ, AMBS_ROOT=root, AMBS_DEVICE=os.environ.get("AMBS_TEST_DEVICE", "cpu"), PYTHONPATH=os.getcwd(), AMBS_REP_PAIRS="8",
           AMBS_REP_STEPS="6", EXP="rep2", BB="tiny")
run = lambda *a: subprocess.run([sys.executable, "-m", *a], env=env, capture_output=True, text=True)
for m in ("spillover", "msrs", "ksteer", "matsteer"):
    for seed in ("42", "43"):
        r = run("ambs.repbase", "train", "--backbone", tiny, "--method", m, "--dtype", "float32", "--seed", seed)
        assert r.returncode == 0, (m, seed, r.stderr[-2500:])
        d = json.load(open(f"{root}/runs/{tiny}/{m}-s{seed}/done.json")); assert d["trainable_params"] > 0, d
        r = run("ambs.repbase", "generate", "--backbone", tiny, "--method", m, "--dtype", "float32", "--seed", seed,
                "--benches", "alpacaeval,harmbench")
        assert r.returncode == 0, (m, seed, r.stderr[-2500:])
        for b in ("alpacaeval", "harmbench"):
            assert len(open(f"{root}/runs/{tiny}/gen/{m}-s{seed}__joint/{b}.jsonl").read().splitlines()) == 3
    if m == "spillover":
        assert d["defining_property_err"] < 1e-6, d
print("all ok")
