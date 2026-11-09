"""GSM8K results of the Qwen models, last 4 layers.

    python scripts/sense_pull_finetuning/qwen_full_summary.py"""
import csv, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "results/analysis/_finetune/final_eval_v2"
J = ROOT / "results/analysis/_finetune"
MODELS = {"qwen1.5-7b": "29-32", "qwen2-7b": "25-28", "qwen2.5-7b": "25-28"}
SEEDS = (0, 1, 2)
rng = np.random.default_rng(0)


def items(name, col):
    p = E / f"gsm8k_full__{name}.csv"
    return np.array([int(r[col]) for r in csv.DictReader(open(p))]) if p.exists() else None


print("Qwen, all GSM8K training data, official GSM8K test set (1,319). ta_pull = separation loss, sign reversed,")
print("last 4 layers. Δ = ta_pull − same-seed baseline; * = 95% bootstrap CI of the paired difference excludes 0.\n")
for m, layers in MODELS.items():
    print(f"=== {m}  (regularised layers {layers})")
    for col, label in (("correct_stop", "stop rule"), ("correct", "without stop rule")):
        pre = items(f"pretrained_{m}", col)
        base = [items(f"final_gsm8k_full_{m}_baseline_s{s}", col) for s in SEEDS]
        pull = [items(f"final_gsm8k_full_{m}_ta_pull_s{s}", col) for s in SEEDS]
        if any(x is None for x in base + pull):
            print(f"  [{label}] missing results")
            continue
        ba, ua = [float(x.mean()) for x in base], [float(x.mean()) for x in pull]
        d = [u - b for u, b in zip(ua, ba)]
        diff = np.concatenate([u - b for u, b in zip(pull, base)])
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        st = "*" if lo > 0 or hi < 0 else " n.s."
        print(f"  [{label}] pretrained {pre.mean():.3f}   baseline {np.mean(ba):.3f} {[round(x, 3) for x in ba]}"
              f"   ta_pull {np.mean(ua):.3f} {[round(x, 3) for x in ua]}" if pre is not None else
              f"  [{label}] baseline {np.mean(ba):.3f}   ta_pull {np.mean(ua):.3f}")
        print(f"             Δ {np.mean(d):+.3f}{st}  per seed {[round(x, 3) for x in d]}  better in "
              f"{sum(x > 0 for x in d)}/3  95% CI [{lo:+.3f}, {hi:+.3f}]")
    for v in ("baseline", "ta_pull"):
        js = [J / f"final_gsm8k_full_{m}_{v}_s{s}.json" for s in SEEDS]
        if all(p.exists() for p in js):
            cs = [json.load(open(p))["geo_after"]["cos_sep"] for p in js]
            print(f"  {v:9s} cos_sep (layers {layers}) {np.mean(cs):+.4f}")
    print()
