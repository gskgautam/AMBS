"""Preference-trained AMBS (ambs/pref.py) on CPU with a tiny Llama: zero offset at initialisation (policy == reference),
DPO loss decreases, an interrupted + resumed run equals an uninterrupted one exactly, validation-accuracy layer
selection, and generation with the saved offsets (joint and single-pathway modes).
PYTHONPATH=. python tests/test_pref.py"""
import json, os, shutil, subprocess, sys, tempfile
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers


def make_root():
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
        W(f"{root}/data/pairs_{o}.jsonl", [{"instruction": f"q{i}", "input": "", "chosen": f"good {i}", "rejected": f"bad {i}"} for i in range(40)])
    for b in ("alpacaeval", "truthfulqa", "beavertails"):
        W(f"{root}/data/eval_{b}.jsonl", [{"id": f"{b}-{i}", "instruction": f"q{i}", "reference": "r"} for i in range(3)])
    return root, tiny


def run(root, *a, extra=None):
    env = dict(os.environ, AMBS_ROOT=root, AMBS_DEVICE="cpu", PYTHONPATH=os.getcwd(), AMBS_PREF_STEPS="120",
               EXP="pref", BB="tiny", **(extra or {}))
    return subprocess.run([sys.executable, "-m", *a], env=env, capture_output=True, text=True)


# zero offset at initialisation => the trained policy starts exactly at the reference policy
sys.path.insert(0, os.getcwd())
from ambs.model import Offsets
m0 = Offsets(32, 100, r=8, n=3, variant="ic", seed=42, b_init="zero")
h = torch.randn(5, 32)
assert all(torch.equal(m0.offset(i, h), torch.zeros_like(h)) for i in range(3))
assert torch.allclose(m0.mixed_offset(h, [0.2, 0.3, 0.5]), sum(w * m0.offset(i, h) for i, w in enumerate([0.2, 0.3, 0.5])))

# uninterrupted reference run
rootA, tinyA = make_root()
r = run(rootA, "ambs.pref", "train", "--backbone", tinyA, "--layer", "mid", "--dtype", "float32")
assert r.returncode == 0, r.stderr[-2000:]
name = "pref-ic-r8-lam0.1-lr0.0002-layermid-s42"
runA = f"{rootA}/runs/{tinyA}/{name}"
log = [json.loads(l) for l in open(f"{runA}/train_log.jsonl")]
assert log[-1]["dpo"] < log[0]["dpo"], [x["dpo"] for x in log]
cfg = json.load(open(f"{runA}/config.json")); assert cfg["layer"] == 2 and cfg["trainable_params"] == 3 * (32 * 8 * 2 + 32)

# interrupted (A100 asked to stop) + resumed run must give identical offsets
rootB, tinyB = make_root()
shutil.rmtree(tinyB); shutil.copytree(tinyA, tinyB)
L = f"{rootB}/locks/pref-tiny"; os.makedirs(L, exist_ok=True); open(f"{L}/takeover", "w").write("x")
r = run(rootB, "ambs.pref", "train", "--backbone", tinyB, "--layer", "mid", "--dtype", "float32", extra={"AMBS_ROLE": "a100"})
runB = f"{rootB}/runs/{tinyB}/{name}"
assert r.returncode == 99 and json.load(open(f"{runB}/ckpt/state.json"))["step"] < 120, (r.returncode, r.stderr[-1500:])
os.remove(f"{L}/takeover")
r = run(rootB, "ambs.pref", "train", "--backbone", tinyB, "--layer", "mid", "--dtype", "float32")
assert r.returncode == 0, r.stderr[-2000:]
a, b = torch.load(f"{runA}/offsets.pt"), torch.load(f"{runB}/offsets.pt")
assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a), {k: (a[k] - b[k]).abs().max().item() for k in a}

# final-layer variant + validation selection
r = run(rootA, "ambs.pref", "train", "--backbone", tinyA, "--layer", "final", "--dtype", "float32")
assert r.returncode == 0, r.stderr[-2000:]
r = run(rootA, "ambs.pref", "valacc", "--backbone", tinyA, "--dtype", "float32")
assert r.returncode == 0, r.stderr[-2000:]
sel = json.load(open(f"{rootA}/runs/{tinyA}/pref_valacc.json"))
assert sel["selected"] in ("mid", "final") and set(sel["valacc"]) == {"mid", "final"}

# generation reads the saved config (layer, variant) through ambs.generate
for mode in ("joint", "path-harm"):
    r = run(rootA, "ambs.generate", "--backbone", tinyA, "--run", name, "--mode", mode, "--benches", "alpacaeval",
            "--batch", "2", "--dtype", "float32")
    assert r.returncode == 0, (mode, r.stderr[-2000:])
    assert len(open(f"{rootA}/runs/{tinyA}/gen/{name}__{mode}/alpacaeval.jsonl").read().splitlines()) == 3
print("pref: zero init, DPO loss decreases, exact stop/resume, layer selection, generation: OK")
