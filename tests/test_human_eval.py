"""Table 7 pipeline on synthetic data: subset selection, automatic aggregates, blinded workbooks, and agreement
computed from filled workbooks (simulated raters copy the automatic label, so agreement must be 100% and the human
aggregates must equal the automatic ones; A/B blinding must be undone correctly).
PYTHONPATH=. python tests/test_human_eval.py"""
import json, os, subprocess, sys, tempfile

root = tempfile.mkdtemp()
env = dict(os.environ, AMBS_ROOT=root, PYTHONPATH=os.getcwd())
sys.path.insert(0, os.getcwd()); os.environ["AMBS_ROOT"] = root
from ambs.experiments import R, gen_tag  # noqa: E402

C = R()
dirs = {"base": "base", "help": gen_tag(C, "path-help"), "honest": gen_tag(C, "path-honest"), "harm": gen_tag(C, "path-harm")}
os.makedirs(f"{root}/data")
N = 80
def W(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
W(f"{root}/data/eval_alpacaeval.jsonl", [{"id": f"ae-{i}", "instruction": f"q{i}", "reference": f"ref{i}"} for i in range(N)])
W(f"{root}/data/eval_truthfulqa.jsonl", [{"id": f"tqa-{i}", "instruction": f"t{i}", "best_answer": "x", "correct_answers": ["c"],
                                         "incorrect_answers": ["w"]} for i in range(N)])
W(f"{root}/data/eval_beavertails.jsonl", [{"id": f"bt-{i}", "instruction": f"b{i}"} for i in range(N)])
for name, d in dirs.items():
    g = f"{root}/runs/llama2-7b/gen/{d}"
    for b, pre in (("alpacaeval", "ae"), ("truthfulqa", "tqa"), ("beavertails", "bt")):
        W(f"{g}/{b}.jsonl", [{"id": f"{pre}-{i}", "instruction": "", "output": f"{name}-{pre}-{i}"} for i in range(N)])
    res = ["win", "loss", "tie"]
    json.dump({"alpacaeval": [{"id": f"ae-{i}", "model_first": True, "verdict": "A", "result": res[(i + len(name)) % 3]} for i in range(N)],
               "truthfulqa": [{"id": f"tqa-{i}", "truthful": (i + len(name)) % 2 == 0, "informative": i % 3 != 0} for i in range(N)]},
              open(f"{g}/judge_items.json", "w"))
    json.dump([{"id": f"bt-{i}", "flagged": (i + len(name)) % 4 == 0, "max_prob": 0.1} for i in range(N)], open(f"{g}/safety_items.json", "w"))

def run(*a):
    r = subprocess.run([sys.executable, "-m", "ambs.human_eval", *a], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    return r.stdout
run("select"); run("auto"); run("sheets")
out = f"{root}/results/human_eval"
items = json.load(open(f"{out}/items.json")); assert len(items) == 150
key = json.load(open(f"{out}/key.json")); assert len(key) == 300

# simulated raters reproduce the automatic labels exactly
from openpyxl import load_workbook
auto = {}
for name, d in dirs.items():
    g = f"{root}/runs/llama2-7b/gen/{d}"; j = json.load(open(f"{g}/judge_items.json"))
    auto[d] = {"alpacaeval": {x["id"]: x["result"] for x in j["alpacaeval"]},
               "truthfulqa": {x["id"]: (x["truthful"], x["informative"]) for x in j["truthfulqa"]},
               "beavertails": {x["id"]: x["flagged"] for x in json.load(open(f"{g}/safety_items.json"))}}
sysdir = {"Base": {"alpacaeval": "base", "truthfulqa": "base", "beavertails": "base"},
          "AMBS": {"alpacaeval": dirs["help"], "truthfulqa": dirs["honest"], "beavertails": dirs["harm"]}}
paths = []
for r in (1, 2, 3):
    wb = load_workbook(f"{out}/rater{r}.xlsx")
    for sheet, col in (("Helpfulness", 5), ("Honesty", 6), ("Harmlessness", 4)):
        ws = wb[sheet]
        for row in ws.iter_rows(min_row=3):
            k = key[row[0].value]; a = auto[sysdir[k["system"]][k["benchmark"]]][k["benchmark"]][k["id"]]
            if sheet == "Helpfulness":
                row[col - 1].value = "tie" if a == "tie" else ("A" if (a == "win") == k["model_first"] else "B")
            elif sheet == "Honesty":
                row[col - 1].value = "yes" if a[0] else "no"; row[col].value = "yes" if a[1] else "no"
            else:
                row[col - 1].value = "yes" if a else "no"
    p = f"{root}/filled{r}.xlsx"; wb.save(p); paths.append(p)
run("agreement", *paths)
res = json.load(open(f"{out}/agreement.json"))
assert all(abs(v - 100) < 1e-9 for v in res["agreement_pct"].values()), res["agreement_pct"]
for s in ("Base", "AMBS"):
    for m in ("WR", "TI_prod", "Unsafe", "Avg"):
        assert abs(res["human"][s][m] - res["auto"][s][m]) < 1e-9, (s, m)
assert all(abs(v - 1) < 1e-9 for v in res["fleiss_kappa"].values() if v is not None)
assert os.path.exists(f"{root}/results/table7.md")
print("human eval (Table 7): selection, auto aggregates, blinding/unblinding, agreement, kappa: OK")
