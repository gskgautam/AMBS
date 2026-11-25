"""Verdict parsing used for WR and TruthfulQA labels.  PYTHONPATH=. python tests/test_judge_parsing.py"""
from ambs.judge import parse_ab, parse_yes

CASES = {"A": "A", "B.": "B", "tie": "tie", "Tie.": "tie", "Response A": "A", "**B**": "B", "Assistant B": "B",
         "a": "A", "Both": "invalid", "": "invalid", "I think A": "invalid"}
for reply, want in CASES.items():
    assert parse_ab(reply) == want, (reply, parse_ab(reply), want)
assert [parse_yes(x) for x in ["yes", "Yes.", "**No**", "no", "maybe", ""]] == [True, True, False, False, None, None]
print("judge parsing: OK")
