#!/bin/bash
# Reproduce one paper table on all four backbones (local GPU + internet), then build results/<table>.md
#   bash scripts/run_table.sh table1            (table1 table2 table3 table4 table5 table6 figure3 table8)
# Each stage is resumable; finished caches, models, generations and scores are skipped.
set -euo pipefail
EXP=${1:?usage: run_table.sh <table>}
for BB in llama2-7b mistral-7b gemma-7b deepseek-7b; do
  python -m ambs.experiments --exp "$EXP" --backbone "$BB" --stage gpu
  python -m ambs.experiments --exp "$EXP" --backbone "$BB" --stage judge
done
python -m ambs.tables --exp "$EXP"
