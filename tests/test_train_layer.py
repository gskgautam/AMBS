"""Layer-l constant offsets (Table 6 early/middle layer, Figure 3 layer sweep) on a tiny random model (CPU):
loss falls, a killed run resumes to an identical model, and generation through the layer hook runs.
PYTHONPATH=. python tests/test_train_layer.py"""
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
PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>").save_pretrained(tiny)
os.makedirs(f"{root}/data")
for pw, n in [("help", 10), ("harm", 6), ("honest", 3)]:
    with open(f"{root}/data/train_{pw}.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"{pw}{i}", "instruction": f"q{i}", "input": "", "output": f"{pw} answer"}) + "\n")
env = dict(os.environ, AMBS_ROOT=root, AMBS_CKPT_EVERY_LAYER="5")
args = [sys.executable, "-m", "ambs.train_layer", "--backbone", tiny, "--layer", "2", "--batch", "2", "--epochs", "30", "--lr", "0.05"]
def go(seed, stop=0):
    return subprocess.run(args + ["--seed", str(seed)], env=dict(env, AMBS_STOP_AFTER=str(stop)), capture_output=True, text=True)
r = go(1); assert r.returncode == 0, r.stderr[-2000:]
name = lambda s: f"{root}/runs/{tiny}/constant-r8-lam0.1-lr0.05-b2-layer2-s{s}"
log = [json.loads(l) for l in open(f"{name(1)}/train_log.jsonl")]
assert sum(v for k, v in log[-1].items() if k.startswith("nll")) < sum(v for k, v in log[0].items() if k.startswith("nll"))
r = go(2, stop=62); assert r.returncode != 0 and os.path.exists(f"{name(2)}/checkpoint.pt")
r = go(2); assert r.returncode == 0 and "resuming" in r.stdout, r.stderr[-2000:]
# identical-resume check: train seed 2 without interruption in a fresh root and compare
import shutil
root2 = tempfile.mkdtemp(); shutil.copytree(f"{root}/data", f"{root2}/data")
r = subprocess.run(args + ["--seed", "2"], env=dict(env, AMBS_ROOT=root2), capture_output=True, text=True); assert r.returncode == 0
x = torch.load(f"{name(2)}/offsets.pt"); y = torch.load(f"{root2}/runs/{tiny}/constant-r8-lam0.1-lr0.05-b2-layer2-s2/offsets.pt")
assert all(torch.allclose(x[k], y[k], atol=1e-6) for k in x), "resumed layer run differs"
# generation through the layer hook
from ambs.model import Offsets, install_hook
cfgj = json.load(open(f"{name(2)}/config.json"))
m = Offsets(cfgj["d"], cfgj["vocab"], r=cfgj["rank"], n=3, variant="constant", seed=2); m.load_state_dict(x)
model = LlamaForCausalLM.from_pretrained(tiny); h = install_hook(model, m, [1/3] * 3, layer=cfgj["layer"])
out = model.generate(torch.tensor([[1, 40, 41]]), max_new_tokens=5, do_sample=True, top_p=0.9, top_k=0); h.remove()
assert out.shape[1] == 8
print("layer-l training: loss falls, kill+resume identical, generation with layer hook: OK")
