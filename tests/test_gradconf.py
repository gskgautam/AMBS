"""F5 gradient-conflict script on synthetic cached states (CPU). PYTHONPATH=. python tests/test_gradconf.py"""
import json, os, subprocess, sys, tempfile
import numpy as np, torch
root = tempfile.mkdtemp(); c = f"{root}/runs/fake/cache"; os.makedirs(c)
d, V = 32, 50; rng = np.random.default_rng(0); W = torch.randn(V, d) * 0.3; torch.save(W, f"{root}/W.pt")
for k, (pw, n) in enumerate([("help", 40), ("harm", 25), ("honest", 7)]):
    off = np.concatenate([[0], np.cumsum(rng.integers(1, 6, n))]).astype(np.int64)
    H = rng.standard_normal((off[-1], d)).astype(np.float32); y = rng.integers(0, V, off[-1]).astype(np.int32)
    np.save(f"{c}/{pw}.h.npy", H.astype(np.float16)); np.save(f"{c}/{pw}.y.npy", y); np.save(f"{c}/{pw}.off.npy", off)
json.dump({"backbone": "x", "d": d, "vocab": V, "pathways": {}}, open(f"{c}/meta.json", "w"))
env = dict(os.environ, AMBS_ROOT=root, AMBS_TEST_W=f"{root}/W.pt")
r = subprocess.run([sys.executable, "-m", "ambs.train", "--backbone", "fake", "--variant", "ic", "--rank", "24", "--scheme", "pooled",
                    "--batch", "4", "--epochs", "3", "--name", "ic-r24-lam0.1-lr0.0002-b16-pooled-s42"], env=env, capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-1500:]
r = subprocess.run([sys.executable, "-m", "ambs.gradconf", "--backbone", "fake", "--n", "5", "--batch", "4"], env=env, capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-1500:]
res = json.load(open(f"{root}/runs/fake/gradconf.json"))
assert set(res) == {"init", "trained"} and all(-1 <= v["mean"] <= 1 for t in res.values() for v in t.values()), res
print("gradient conflict: init and trained cosines computed: OK")
