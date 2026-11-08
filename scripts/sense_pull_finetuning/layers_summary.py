"""GSM8K results for the last 4 / 8 / 10 layers: mean +/- SD, seeds better, t-test, Wilcoxon, cos_sep.

    python scripts/sense_pull_finetuning/layers_summary.py"""
import csv, json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "results/analysis/_finetune/final_eval_v2"
J = ROOT / "results/analysis/_finetune"
N_LAYERS = {"llama-3-8b": 32, "llama-3.1-8b": 32, "qwen1.5-7b": 32, "qwen2-7b": 28, "qwen2.5-7b": 28}
SEEDS = (0, 1, 2)


def acc(name):
    p = E / f"gsm8k_full__{name}.csv"
    return float(np.mean([int(r["correct_stop"]) for r in csv.DictReader(open(p))])) * 100 if p.exists() else None


def cos_sep(name):
    p = J / f"{name}.json"
    return json.load(open(p))["geo_after"]["cos_sep"] if p.exists() else None


sd = lambda x: np.std(x, ddof=1) if len(x) > 1 else float("nan")
print("GSM8K, all training data, official test set, stop rule; ta_pull on the last k layers; 3 seeds; values in %.\n")
print(f"{'model':13s} {'layers':>9s}  {'accuracy mean±SD':>17s}  {'Δ vs baseline':>14s}  {'better':>6s}  "
      f"{'t-test p':>8s}  {'Wilcoxon p':>10s}  {'cos_sep':>8s}")
for m, n in N_LAYERS.items():
    base = [acc(f"final_gsm8k_full_{m}_baseline_s{s}") for s in SEEDS]
    if any(b is None for b in base):
        print(f"{m:13s}  missing baseline")
        continue
    print(f"{m:13s} {'baseline':>9s}  {np.mean(base):7.1f} ± {sd(base):4.1f}")
    for k in (4, 8, 10):
        v = "ta_pull" if k == 4 else f"ta_pull_last{k}"
        arr = [acc(f"final_gsm8k_full_{m}_{v}_s{s}") for s in SEEDS]
        lay = f"{n - k + 1}-{n}"
        if any(a is None for a in arr):
            print(f"{'':13s} {lay:>9s}  missing")
            continue
        d = np.array(arr) - np.array(base)
        tp = stats.ttest_1samp(d, 0).pvalue
        wp = stats.wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
        cs = [cos_sep(f"final_gsm8k_full_{m}_{v}_s{s}") for s in SEEDS]
        cs = f"{np.mean(cs):+.3f}" if all(c is not None for c in cs) else "-"
        print(f"{'':13s} {lay:>9s}  {np.mean(arr):7.1f} ± {sd(arr):4.1f}   {d.mean():+5.1f} ± {sd(d):4.1f}   "
              f"{int((d > 0).sum())}/3    {tp:8.2f}  {wp:10.2f}  {cs:>8s}   per seed {' '.join(f'{x:+.1f}' for x in d)}")
    print()
