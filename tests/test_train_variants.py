"""Every model variant and training scheme trains (loss falls) on synthetic cached hidden states; frozen-b stays frozen.
PYTHONPATH=. python tests/test_train_variants.py"""
import glob, json, os, subprocess, sys, tempfile
import numpy as np
import torch

root = tempfile.mkdtemp()
c = f"{root}/runs/fake/cache"; os.makedirs(c)
d, V = 32, 50
rng = np.random.default_rng(0)
W = torch.randn(V, d) * 0.3
torch.save(W, f"{root}/W.pt")
for k, (pw, n) in enumerate([("help", 40), ("harm", 25), ("honest", 7)]):
    off = np.concatenate([[0], np.cumsum(rng.integers(1, 6, n))]).astype(np.int64)
    H = rng.standard_normal((off[-1], d)).astype(np.float32)
    y = ((H @ W.numpy().T) + 5 * np.eye(V)[k * 10][None]).argmax(1).astype(np.int32)
    np.save(f"{c}/{pw}.h.npy", H.astype(np.float16)); np.save(f"{c}/{pw}.y.npy", y); np.save(f"{c}/{pw}.off.npy", off)
json.dump({"backbone": "x", "d": d, "vocab": V, "pathways": {}}, open(f"{c}/meta.json", "w"))
env = dict(os.environ, AMBS_ROOT=root, AMBS_TEST_W=f"{root}/W.pt")
CASES = ["--variant constant", "--variant ic", "--variant no_shared_ref", "--variant direct_vector --wd 0",
         "--variant logit_bias --lam 0", "--variant direct_update", "--variant constant --rank 0",
         "--variant constant --b_init zero", "--variant constant --freeze_b", "--variant constant --scheme sequential",
         "--variant constant --scheme pooled", "--variant direct_vector --wd 0 --scheme pooled",
         "--variant constant --scheme shared_labels", "--variant constant --lam 0"]
for i, args in enumerate(CASES):
    out = subprocess.run([sys.executable, "-m", "ambs.train", "--backbone", "fake", "--batch", "4", "--epochs", "20",
                          "--lr", "3e-2", "--name", f"case{i}"] + args.split(), env=env, capture_output=True, text=True)
    assert out.returncode == 0, (args, out.stderr[-2000:])
    log = [json.loads(l) for l in open(f"{root}/runs/fake/case{i}/train_log.jsonl")]
    first_stage = [r for r in log if r["stage"] == 0]
    tot = lambda r: sum(v for k, v in r.items() if k.startswith("nll"))
    assert tot(first_stage[-1]) < tot(first_stage[0]), (args, tot(first_stage[0]), tot(first_stage[-1]))
    print(f"ok  {args}")
from ambs.model import Offsets
fb = torch.load(f"{root}/runs/fake/case8/offsets.pt")
assert torch.equal(fb["b"], Offsets(d, V, r=8, n=3, variant="constant", seed=42).b.data), "freeze_b changed b"
print("all training variants: OK")
