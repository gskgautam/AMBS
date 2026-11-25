"""A run killed mid-training and resumed from its checkpoint must end with exactly the same model and log as an
uninterrupted run (parallel and sequential schemes).  PYTHONPATH=. python tests/test_checkpoint_resume.py"""
import json, os, subprocess, sys, tempfile
import numpy as np
import torch

root = tempfile.mkdtemp()
c = f"{root}/runs/fake/cache"; os.makedirs(c)
d, V = 16, 30
rng = np.random.default_rng(1)
torch.save(torch.randn(V, d) * 0.3, f"{root}/W.pt")
for pw, n in [("help", 30), ("harm", 20), ("honest", 6)]:
    off = np.concatenate([[0], np.cumsum(rng.integers(1, 5, n))]).astype(np.int64)
    np.save(f"{c}/{pw}.h.npy", rng.standard_normal((off[-1], d)).astype(np.float16))
    np.save(f"{c}/{pw}.y.npy", rng.integers(0, V, off[-1]).astype(np.int32)); np.save(f"{c}/{pw}.off.npy", off)
json.dump({"backbone": "x", "d": d, "vocab": V, "pathways": {}}, open(f"{c}/meta.json", "w"))
base_env = dict(os.environ, AMBS_ROOT=root, AMBS_TEST_W=f"{root}/W.pt", AMBS_CKPT_EVERY="7")

def run(name, extra, stop=0):
    env = dict(base_env, AMBS_STOP_AFTER=str(stop))
    return subprocess.run([sys.executable, "-m", "ambs.train", "--backbone", "fake", "--batch", "4", "--epochs", "6",
                           "--name", name] + extra, env=env, capture_output=True, text=True)

for scheme, stop in ((["--variant", "ic"], 23), (["--variant", "constant", "--scheme", "sequential"], 23),
                     (["--variant", "constant", "--scheme", "sequential", "--seed", "7"], 50)):
    tag = scheme[-1] + f"-stop{stop}"
    assert run(f"full-{tag}", scheme).returncode == 0
    r = run(f"kill-{tag}", scheme, stop=stop)
    assert r.returncode != 0 and os.path.exists(f"{root}/runs/fake/kill-{tag}/checkpoint.pt"), r.stderr[-500:]
    r = run(f"kill-{tag}", scheme)
    assert r.returncode == 0 and "resuming from checkpoint" in r.stdout, r.stderr[-1500:]
    a = torch.load(f"{root}/runs/fake/full-{tag}/offsets.pt"); b = torch.load(f"{root}/runs/fake/kill-{tag}/offsets.pt")
    assert all(torch.equal(a[k], b[k]) for k in a), f"{tag}: resumed model differs"
    la = [json.loads(l) for l in open(f"{root}/runs/fake/full-{tag}/train_log.jsonl")]
    lb = [json.loads(l) for l in open(f"{root}/runs/fake/kill-{tag}/train_log.jsonl")]
    strip = lambda L: [{k: v for k, v in x.items() if k != "sec"} for x in L]
    assert strip(la) == strip(lb), f"{tag}: logs differ"
    assert not os.path.exists(f"{root}/runs/fake/kill-{tag}/checkpoint.pt")
    print(f"ok  {tag}: killed at global step {stop}, resumed -> identical model and log")
print("checkpoint resume: OK")
