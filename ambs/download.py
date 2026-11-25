"""Download models and raw datasets into HF_HOME (needs internet).

Usage: python -m ambs.download [--models all|name,...] [--datasets]
"""
import argparse
import os
import sys

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from .common import ALL_BACKBONES as BACKBONES, BEAVER_DAM, DATA, SECRETS

EXTRA_MODELS = {"harmbench-cls": "cais/HarmBench-Llama-2-13b-cls", "llamaguard3": "meta-llama/Llama-Guard-3-8B"}
HARMBENCH_CSV = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/"
                 "harmbench_behaviors_text_test.csv")

DATASETS = [
    "tatsu-lab/alpaca",
    "PKU-Alignment/BeaverTails",
    "truthfulqa/truthful_qa",
]


def token():
    p = os.path.join(SECRETS, "hf_token")
    if os.path.exists(p):
        return open(p).read().strip()
    return None


def get_model(repo):
    api = HfApi(token=token())
    files = api.list_repo_files(repo)
    has_st = any(f.endswith(".safetensors") for f in files)
    patterns = ["*.json", "tokenizer*", "*.model", "*.tiktoken"]
    patterns += ["*.safetensors"] if has_st else ["*.bin"]
    path = snapshot_download(repo, allow_patterns=patterns, token=token())
    print(f"OK model {repo} -> {path}", flush=True)


def get_datasets():
    for repo in DATASETS:
        path = snapshot_download(repo, repo_type="dataset", token=token())
        print(f"OK dataset {repo} -> {path}", flush=True)
    p = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset", token=token())
    print(f"OK alpaca_eval -> {p}", flush=True)
    import urllib.request
    os.makedirs(os.path.join(DATA, "raw"), exist_ok=True)
    dst = os.path.join(DATA, "raw", "harmbench_behaviors_text_test.csv")
    urllib.request.urlretrieve(HARMBENCH_CSV, dst)
    print(f"OK harmbench csv -> {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--datasets", action="store_true")
    a = ap.parse_args()
    failed = []
    if a.datasets:
        try:
            get_datasets()
        except Exception as e:  # keep going so one failure doesn't block the rest
            print(f"FAIL datasets: {e!r}", flush=True)
            failed.append("datasets")
    names = []
    if a.models == "all":
        names = list(BACKBONES) + ["beaver-dam"]
    elif a.models:
        names = a.models.split(",")
    for n in names:
        repo = BEAVER_DAM if n == "beaver-dam" else EXTRA_MODELS.get(n) or BACKBONES[n]
        try:
            get_model(repo)
        except Exception as e:
            print(f"FAIL model {repo}: {e!r}", flush=True)
            failed.append(repo)
    print("FAILED:", failed if failed else "none", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
