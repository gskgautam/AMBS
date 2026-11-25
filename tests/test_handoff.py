"""A100 -> H200 handoff (ambs/handoff.py), on CPU with a fake `qstat`/`qdel`:
  1. ownership rules (A100 yields to a running H200, H200 takes over a dead A100 owner, 'done' stops both);
  2. training: an A100 run asked to stop saves a checkpoint and exits 99; the H200 run resumes to exactly the model
     of an uninterrupted run;
  3. generation: an A100 run asked to stop keeps its finished samples; the H200 run generates only the rest.
PYTHONPATH=. python tests/test_handoff.py"""
import json, os, subprocess, sys, tempfile
import numpy as np
import torch

root = tempfile.mkdtemp(); bindir = f"{root}/bin"; os.makedirs(bindir)
# fake PBS: job ids listed in $root/running are 'R', everything else unknown; qdel is a no-op logger
open(f"{bindir}/qstat", "w").write(f'#!/bin/bash\nid="${{@: -1}}"; grep -qx "$id" {root}/running 2>/dev/null && echo "    job_state = R"; exit 0\n')
open(f"{bindir}/qdel", "w").write(f'#!/bin/bash\necho "$@" >> {root}/qdel.log\n')
os.chmod(f"{bindir}/qstat", 0o755); os.chmod(f"{bindir}/qdel", 0o755)
open(f"{root}/running", "w").write("")
env = dict(os.environ, AMBS_ROOT=root, PATH=f"{bindir}:{os.environ['PATH']}", EXP="tX", BB="fake", PYTHONPATH=os.getcwd())
L = f"{root}/locks/tX-fake"

def hand(cmd, role, job):
    return subprocess.run([sys.executable, "-m", "ambs.handoff", cmd, "--role", role], capture_output=True, text=True,
                          env=dict(env, AMBS_ROLE=role, PBS_JOBID=job)).returncode

# 1. ownership
assert hand("acquire", "a100", "A1") == 0 and open(f"{L}/owner").read().split() == ["a100", "A1"]
open(f"{root}/running", "w").write("H1\n")
os.remove(f"{L}/owner"); assert hand("acquire", "h200", "H1") == 0          # H200 owns
assert hand("acquire", "a100", "A2") == 3                                     # A100 yields to a running H200
open(f"{root}/running", "w").write("")                                       # H200 died: stale owner
assert hand("acquire", "a100", "A2") == 0
assert hand("acquire", "h200", "H2") == 0 and open(f"{L}/owner").read().split()[0] == "h200"  # A2 not running -> take
hand("finish", "h200", "H2"); assert hand("acquire", "a100", "A3") == 3 and hand("acquire", "h200", "H3") == 3
os.remove(f"{L}/done")
open(f"{L}/owner", "w").write("a100 A7\n")   # another job owns the stage: an A100 job must stop
chk = lambda job: subprocess.run([sys.executable, "-c", "from ambs.handoff import should_stop; print(should_stop())"],
                                 env=dict(env, AMBS_ROLE="a100", PBS_JOBID=job), capture_output=True, text=True).stdout.strip()
assert chk("A9") == "True" and chk("A7") == "False"
os.remove(f"{L}/owner")
print("ok  ownership rules (incl. simultaneous start)")

# 2. training handoff (reuses the synthetic cache of test_checkpoint_resume)
c = f"{root}/runs/fake/cache"; os.makedirs(c); d, V = 16, 30; rng = np.random.default_rng(1)
torch.save(torch.randn(V, d) * 0.3, f"{root}/W.pt")
for pw, n in [("help", 60), ("harm", 40), ("honest", 12)]:
    off = np.concatenate([[0], np.cumsum(rng.integers(1, 5, n))]).astype(np.int64)
    np.save(f"{c}/{pw}.h.npy", rng.standard_normal((off[-1], d)).astype(np.float16))
    np.save(f"{c}/{pw}.y.npy", rng.integers(0, V, off[-1]).astype(np.int32)); np.save(f"{c}/{pw}.off.npy", off)
json.dump({"backbone": "x", "d": d, "vocab": V, "pathways": {}}, open(f"{c}/meta.json", "w"))
tenv = dict(env, AMBS_TEST_W=f"{root}/W.pt")
def train(name, role):
    return subprocess.run([sys.executable, "-m", "ambs.train", "--backbone", "fake", "--batch", "2", "--epochs", "4",
                           "--name", name, "--variant", "ic"], env=dict(tenv, AMBS_ROLE=role), capture_output=True, text=True)
assert train("full", "h200").returncode == 0
os.makedirs(L, exist_ok=True); open(f"{L}/takeover", "w").write("H9\n")
r = train("handed", "a100"); assert r.returncode == 99 and "HANDOFF" in r.stdout, (r.returncode, r.stderr[-800:])
assert os.path.exists(f"{root}/runs/fake/handed/checkpoint.pt")
os.remove(f"{L}/takeover")
r = train("handed", "h200"); assert r.returncode == 0 and "resuming from checkpoint" in r.stdout, r.stderr[-800:]
a = torch.load(f"{root}/runs/fake/full/offsets.pt"); b = torch.load(f"{root}/runs/fake/handed/offsets.pt")
assert all(torch.equal(a[k], b[k]) for k in a), "handed-over training differs from uninterrupted training"
print("ok  training: A100 stops at step 50 with a checkpoint, H200 resumes to the identical model")

# 3. generation handoff on a tiny random Llama
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
tiny = f"{root}/tiny"
cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=4, bos_token_id=1, eos_token_id=2)
torch.manual_seed(0); LlamaForCausalLM(cfg).save_pretrained(tiny)
vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, **{chr(i): i - 29 for i in range(32, 127)}}
tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Split("", "isolated")
PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                        model_input_names=["input_ids", "attention_mask"]).save_pretrained(tiny)
os.makedirs(f"{root}/data", exist_ok=True)
with open(f"{root}/data/eval_alpacaeval.jsonl", "w") as f:
    for i in range(60):
        f.write(json.dumps({"id": f"ae-{i}", "instruction": "q" * (i % 9 + 1), "reference": "r"}) + "\n")
genv = dict(env, AMBS_DEVICE="cpu")
def gen(role):
    return subprocess.run([sys.executable, "-m", "ambs.generate", "--backbone", tiny, "--mode", "base", "--benches",
                           "alpacaeval", "--batch", "3", "--dtype", "float32"], env=dict(genv, AMBS_ROLE=role),
                          capture_output=True, text=True, timeout=600)
out = f"{root}/runs/{tiny}/gen/base"
import threading, time
def request_takeover():  # the H200 job starts while the A100 is generating
    t0 = time.time()
    while not os.path.exists(f"{out}/alpacaeval.partial.jsonl") and time.time() - t0 < 300:
        time.sleep(0.005)
    open(f"{L}/takeover", "w").write("H9\n")
th = threading.Thread(target=request_takeover); th.start()
r = gen("a100"); th.join(); assert r.returncode == 99, (r.returncode, r.stderr[-800:])
part = [json.loads(l) for l in open(f"{out}/alpacaeval.partial.jsonl")]
assert 3 <= len(part) < 60 and len(part) % 3 == 0, len(part)
assert not os.path.exists(f"{out}/alpacaeval.jsonl")
os.remove(f"{L}/takeover")
r = gen("h200"); assert r.returncode == 0 and f"resuming, {len(part)}/60" in r.stdout, r.stderr[-800:]
final = [json.loads(l) for l in open(f"{out}/alpacaeval.jsonl")]
assert len(final) == 60 and [x["id"] for x in final] == [f"ae-{i}" for i in range(60)]
kept = {p["id"]: p["raw_output"] for p in part}
assert all(x["raw_output"] == kept[x["id"]] for x in final if x["id"] in kept)
assert not os.path.exists(f"{out}/alpacaeval.partial.jsonl")
# a takeover requested before a benchmark starts stops immediately, with nothing lost
open(f"{L}/takeover", "w").write("H9\n"); os.remove(f"{out}/alpacaeval.jsonl")
r = gen("a100"); assert r.returncode == 99 and not os.path.exists(f"{out}/alpacaeval.partial.jsonl")
os.remove(f"{L}/takeover")
print(f"ok  generation: A100 stops after {len(part)}/60 samples, H200 generates the rest; order and kept samples preserved")
print("handoff: OK")
