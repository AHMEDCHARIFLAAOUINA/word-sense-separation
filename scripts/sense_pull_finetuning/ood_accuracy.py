"""Loaders for MATH (plain-number answers) and Natural Questions, used by eval_final.py."""
import os
import ast, json, random, re
from pathlib import Path

OC = Path(os.environ.get("WSS_BENCH_DATA", "benchmark_data"))


def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


def load_math(n):
    items = json.load(open(OC / "math/math.json", encoding="utf-8"))
    rows = []
    for key in sorted(items):
        r = items[key]
        a = last_boxed(r["solution"])
        if a is None:
            continue
        a = a.replace("\\!", "").replace(",", "").replace(" ", "").replace("$", "").strip()
        if re.fullmatch(r"-?\d+(\.\d+)?", a):
            rows.append({"prompt": f"Question: {r['problem']}\nAnswer:", "gold": [a],
                         "level": r["level"]})
    print(f"MATH: {len(items)} problems, {len(rows)} with a plain-number answer", flush=True)
    random.Random(0).shuffle(rows)
    return rows[:n] if n else rows


def load_nq(n):
    rows = []
    with open(OC / "nq/nq-test.qa.csv", encoding="utf-8") as f:
        for line in f:
            q, a = line.rstrip("\n").split("\t", 1)
            rows.append({"prompt": f"Question: {q.strip()}?\nAnswer:" if not q.strip().endswith("?")
                         else f"Question: {q.strip()}\nAnswer:",
                         "gold": [g for g in ast.literal_eval(a) if g.strip()], "level": ""})
    print(f"NQ: {len(rows)} questions", flush=True)
    return rows[:n] if n else rows
