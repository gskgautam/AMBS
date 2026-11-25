# A Controlled Study of Objective-Specific Readout Steering for Multi-Objective LLM Alignment

## Repository Map

```

AMBS/
├── ambs/                    # All Code (Run as `python -m ambs.<module>`)
│   ├── common.py            # Paths, Backbones, Prompt Template, Constants
│   ├── download.py          # Backbones, Classifiers and Raw Datasets
│   ├── prepare_data.py      # Train/Eval Splits, Label Perturbation, HarmBench Test Cases
│   ├── cache.py             # Frozen-Backbone Readout States (Computed Once per Backbone)
│   ├── model.py             # AMBS, AMBS-IC, SA, DV and Every Ablation Variant; Generation Hooks
│   ├── train.py             # Readout Training (Parallel, Sequential, Shared-Label, Pooled Schemes)
│   ├── train_layer.py       # Offsets at an Intermediate Layer
│   ├── generate.py          # Decoding: Base, Single Pathway, Separate, Pooled Joint
│   ├── judge.py             # GPT-4o / Claude Judging (WR, TI_prod), Batch API, Spending Cap
│   ├── safety.py            # Beaver-Dam-7B (SS) and Llama Guard 3
│   ├── harmbench.py         # HarmBench ASR with the Official Classifier
│   ├── baselines.py         # Matched CAA, LoRA-SFT and DPO
│   ├── repbase.py           # CAA (Tuned), ITI, LoReFT, MAT-Steer, K-Steering, MSRS, Spillover
│   ├── targets.py           # Supervision-Target Quality (Stronger and Self-Distilled Targets)
│   ├── pref.py              # Preference-Trained AMBS
│   ├── evals.py             # On-Topic Safety, Capability, OOD and Deployment Checks
│   ├── heldout.py           # Held-Out and Validation Loss
│   ├── gradconf.py          # Gradient Conflict Between Pathways
│   ├── human_eval.py        # Human Audit: Sample Selection, Rater Workbooks, Agreement
│   ├── handoff.py           # Optional Job Hand-Off for PBS Clusters (No-Op on a Local Machine)
│   ├── experiments.py       # Experiment Name -> Runs (Stages: GPU, Judge, List)
│   ├── tables.py            # Core Result Tables
│   ├── extra.py             # Analysis Tables (Significance, Interference, Seeds, Controllability, ...)
│   ├── freeanal.py          # Analyses Without GPU or API Calls (Geometry, Depth CIs, Length/Refusal)
│   └── assets/              # HarmBench Classifier Prompt
├── scripts/run_table.sh     # One Experiment on All Four Backbones, Then Its Table
├── tests/                   # CPU Tests (Tiny Random Models)
├── data/                    # Fixed Train/Eval Splits Used in the Paper (See Section 3)
├── requirements.txt
└── README.md

```

All outputs are written under the repository root by default (`runs/<backbone>/`, `results/`). Set `AMBS_ROOT` to use another working directory.

## 1. Prepare the Environment

```bash
cd AMBS                      # the cloned repository
python3.11 -m venv env && source env/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
export HF_HOME=$PWD/hf_home
```

The experiments used one NVIDIA H200 or A100 (80 GB) per job, FP16 weights (BF16 if a backbone overflows; the switch is recorded in `runs/<backbone>/DTYPE`).

## 2. Add API Keys

Keys are read from plain files in `secrets/`, which is git-ignored. Never commit them.

```bash
mkdir -p secrets && chmod 700 secrets
printf '%s' "<your Hugging Face token>" > secrets/hf_token
printf '%s' "<your OpenAI key>"         > secrets/openai_key
printf '%s' "<your Anthropic key>"      > secrets/anthropic_key   # optional: Claude judge in Table I
chmod 600 secrets/*
```

Llama-2-7B, Mistral-7B-v0.1, Gemma-7B and Llama Guard 3 are gated on Hugging Face; accept their licences first. Judging spend is capped in `ambs/judge.py` (`AMBS_API_CAP_USD` overrides it).

## 3. Datasets

The exact splits used in the paper are included in `data/` (37 MB), so no preparation step is needed.

```
data/
├── train_{help,harm,honest}.jsonl            # Training Data per Objective
├── train_{help,harm,honest}.val.jsonl        # Validation Splits (Model Selection)
├── train_{help,harm,honest}.heldout.jsonl    # Held-Out Splits (Loss Analysis)
├── train_{help,harm,honest}.pert.jsonl       # Label-Perturbed Training Data (Controls)
├── eval_{alpacaeval,beavertails,truthfulqa,harmbench}.jsonl
├── subset_{alpacaeval,truthfulqa}_200.json   # Fixed 200-Prompt Subsets (Seed 42)
├── ext/                                      # MMLU, GSM8K, XSTest, MT-Bench, AdvBench, HaluEval
├── raw/harmbench_behaviors_text_test.csv
└── data_stats.json                           # Split Sizes, Overlap Removal, Perturbation Counts
```

Download the backbones and classifiers:

```bash
python -m ambs.download --models all                         # 4 backbones + Beaver-Dam-7B
python -m ambs.download --models harmbench-cls,llamaguard3   # safety classifiers
```

The files are redistributed under their original licences: Alpaca, AlpacaEval and BeaverTails (CC BY-NC 4.0), TruthfulQA and MT-Bench (Apache 2.0), HarmBench, MMLU, GSM8K, AdvBench and HaluEval (MIT), XSTest (CC BY 4.0). Because of the CC BY-NC sources, `data/` is for non-commercial research use. BeaverTails, HarmBench and AdvBench contain harmful prompts and responses.

> **Note.** The preference pairs (`pairs_{help,harm,honest}.jsonl`) and the stronger GPT-4o targets (`train_*.strong.jsonl`) are not included in `data/`. The DPO reference (Fig. 3(a)), the preference-trained AMBS and the target-quality runs (Fig. 6(a)) need them, so build them first with `python -m ambs.targets build`, which calls the OpenAI API and needs `secrets/openai_key`.

To rebuild everything from the original sources (this overwrites `data/`):

```bash
python -m ambs.download --datasets
python -m ambs.prepare_data && python -m ambs.evals fetch && python -m ambs.heldout prep
python -m ambs.targets build
```

## 4. Run the Tests (CPU, before any GPU job)

```bash
for t in tests/test_*.py; do python "$t" || break; done
```

## 5. Run an Experiment

Every experiment has a GPU stage (cache, train, generate, local classifiers) and a judge stage (LLM judges, needs
internet). Both are resumable: finished caches, models, generations and scores are skipped.

Backbones: `llama2-7b mistral-7b gemma-7b deepseek-7b`.

```bash
python -m ambs.experiments --exp table1 --backbone llama2-7b --stage list    # show the runs
python -m ambs.experiments --exp table1 --backbone llama2-7b --stage gpu
python -m ambs.experiments --exp table1 --backbone llama2-7b --stage judge   # GPT-4o Batch API (--realtime for the live API)
```

Or all four backbones and the table in one command:

```bash
bash scripts/run_table.sh table1
```

Run the readout cache once per backbone before the first GPU stage (the GPU stage also does it if missing):

```bash
python -m ambs.cache --backbone llama2-7b
```

## 6. Metric Settings

- **WR**: GPT-4o (`gpt-4o-2024-08-06`, temperature 0) compares the answer with the AlpacaEval reference; the A/B
  order is randomised per item (seed 42). Ties count in n, not as wins.
- **TI_prod** = 100 · T̄ · Ī, from GPT-4o truthfulness and informativeness labels.
- **Safe** = 100 − SS, where SS is the Beaver-Dam-7B unsafe rate at threshold 0.5.
- **S_HHH** = (WR + TI_prod + Safe) / 3. A_joint uses the pooled decoder on all three benchmarks; A_sep uses each
  pathway on its own benchmark.
- **ASR**: HarmBench attack success rate. Lower is better for SS and ASR; higher is better for all other metrics.
- Budget runs (seeds, controllability, mixture loss, depth) are judged on fixed 200-prompt subsets (seed 42) of
  AlpacaEval and TruthfulQA; safety always uses the full set.

Every verdict is cached per item (`judge_items.json`), so scores can be re-derived without new API calls.

## 7. Reproduce Fig. 3 

| Panel | Experiments (`--exp`) | Table builder |
|---|---|---|
| (a) S_HHH and compute | `table1`, `mech`, `rep`, `rep2`, `baselines` (DPO needs `python -m ambs.targets build` first) | `python -m ambs.tables --exp table1`, `python -m ambs.extra mech` |
| (b) Older steering methods | `rep` | `python -m ambs.extra rep` |
| (c) Efficiency | `efficiency` | `python -m ambs.extra bench --backbone llama2-7b`, `python -m ambs.extra efficiency` |
| (d) Paired significance | `table1` | `python -m ambs.extra significance` |
| (e) Recent steering methods | `rep2`, `ksfix` | `python -m ambs.extra rep` |
| (f) Seeds | `table2`, `seeds`, `rep2seeds` | `python -m ambs.tables --exp table2`, `python -m ambs.extra seeds` |

## 8. Reproduce Fig. 4 

| Panel | Experiments (`--exp`) | Table builder |
|---|---|---|
| (a) Decomposition vs SA, mixture loss | `mech`, `mechmid`, `mechfull`, `e2` (γ = 1), `e2w` (γ = 0.5, 2), `e3` | `python -m ambs.extra mech` |
| (b) Transfer and cost | `deploy` | `python -m ambs.extra deploy` |
| (c) Controllability (re-weighting) | `ctrl` | `python -m ambs.extra ctrl` |

## 9. Reproduce Fig. 5 

| Panel | Experiments (`--exp`) | Table builder |
|---|---|---|
| Vocabulary-bias control | `table8` | `python -m ambs.tables --exp table8` |
| Interference | `interference`, `interf0` | `python -m ambs.extra interference`, `python -m ambs.extra interf0` |
| Coupling strength λ | `iclam` | `python -m ambs.freeanal` (`lam_geometry`) |
| Gradient conflict | `gradconf` | `python -m ambs.freeanal` (`gradconf`) |
| Pathway geometry | `mech` | `python -m ambs.freeanal` (`geometry`) |
| Training / held-out loss | `heldout` | `results/` JSON from `ambs.heldout` |
| Controls (DV, perturbation, pooled) | `table3` | `python -m ambs.tables --exp table3` |
| Depth under joint decoding | `mid_joint`, `depthval` | `python -m ambs.extra depth`, `python -m ambs.freeanal` (`depth_ci`, `depth_select`) |
| Ablations and sweeps | `table6`, `figure3` | `python -m ambs.tables --exp table6`, `python -m ambs.tables --exp figure3` |

## 10. Reproduce Fig. 6 

Run `python -m ambs.targets build` before panel (a) (see the note in Section 3).

| Panel | Experiments (`--exp`) | Table builder |
|---|---|---|
| (a) Target quality | `targets`, `selfdistill`, `preftest` | `python -m ambs.extra targets` |
| (b) On-topic safety | `ontopic`, `lg3` | `python -m ambs.extra ontopic`, `python -m ambs.freeanal` (`lg3_safe`) |
| (c) HarmBench ASR | `table5`, `harmaffine`, `rep2` | `python -m ambs.tables --exp table5` |
| (d) Instruction-tuned backbones | `table1` with `--backbone llama2-7b-chat` (and the other chat backbones) | `python -m ambs.extra instruct` |

Capability and OOD checks: `--exp cap`, `--exp ood`, then `python -m ambs.extra capability` and `python -m ambs.extra ood`.

## 11. Reproduce Table I 

```bash
for BB in llama2-7b mistral-7b gemma-7b deepseek-7b; do
  python -m ambs.experiments --exp table4 --backbone $BB --stage judge   # GPT-4o + Claude, 3 prompts, Llama Guard 3
done
python -m ambs.tables --exp table4
```

## 12. Reproduce Tables II-III 

```bash
python -m ambs.human_eval select     # fixed 150-sample LLaMA-2-7B subset (50 per benchmark, seed 42)
python -m ambs.human_eval auto       # automatic labels on the subset
python -m ambs.human_eval sheets     # blinded rater workbooks (one per rater)
python -m ambs.human_eval agreement rater1.xlsx rater2.xlsx rater3.xlsx   # agreement, kappa, Tables II-III
```

All result tables are written to `results/` as Markdown and CSV. To rebuild every core table at once run `python -m ambs.tables`, and `python -m ambs.extra new` for the analysis tables.
