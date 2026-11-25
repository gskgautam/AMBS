"""Shared paths, constants and prompt formatting for the AMBS replication."""
import json
import os
import random

import numpy as np

ROOT = os.environ.get("AMBS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
RUNS = os.path.join(ROOT, "runs")
SECRETS = os.path.join(ROOT, "secrets")

BACKBONES = {
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "gemma-7b": "google/gemma-7b",
    "deepseek-7b": "deepseek-ai/deepseek-llm-7b-base",
}
# Instruction-tuned backbones (supplementary): not part of the paper's main tables.
CHAT_BACKBONES = {
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",  # instruction-tuned counterparts of the base backbones
    "gemma-7b-it": "google/gemma-7b-it",
    "deepseek-7b-chat": "deepseek-ai/deepseek-llm-7b-chat",
}
ALL_BACKBONES = {**BACKBONES, **CHAT_BACKBONES}
BEAVER_DAM = "PKU-Alignment/beaver-dam-7b"

# Pathway order is fixed everywhere: 0=help, 1=harm (safe behaviour), 2=honest
PATHWAYS = ["help", "harm", "honest"]
# Separate evaluation: which pathway is used for which benchmark
SEP_PATHWAY = {"alpacaeval": "help", "beavertails": "harm", "truthfulqa": "honest", "harmbench": "harm"}

MAX_LEN = 512
MAX_NEW_TOKENS = 256

PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)
PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)


def format_prompt(instruction, inp=""):
    if inp and inp.strip():
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=inp)
    return PROMPT_NO_INPUT.format(instruction=instruction)


def build_prompt(bb, tok, instruction, inp=""):
    """Prompt string, whether the tokenizer should add special tokens, and the prefix that starts a response.

    Base backbones: the paper's Alpaca template. Instruction-tuned backbones: the model's own chat template
    (it already contains BOS, so no special tokens are added; Llama-2-chat responses start with a space after [/INST]).
    """
    if bb in CHAT_BACKBONES:
        content = instruction + (f"\n\n{inp}" if inp and inp.strip() else "")
        prompt = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
        return prompt, False, " "
    return format_prompt(instruction, inp), True, ""


def clean_generation(text):
    """Base models often continue with a new '### Instruction' block; cut there (pre-registered)."""
    for stop in ["\n### ", "### Instruction", "\n\n\n\n"]:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def seed_all(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _tmp(path):
    """Temporary name unique to this process, so concurrent jobs never write into the same file."""
    return f"{path}.{os.uname().nodename}.{os.getpid()}.tmp"


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
