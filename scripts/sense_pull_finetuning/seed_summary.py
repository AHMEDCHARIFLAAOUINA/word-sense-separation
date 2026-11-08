"""GSM8K results per seed with standard deviations (llama-3-8b with 10 seeds).

    python scripts/sense_pull_finetuning/seed_summary.py"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "results/analysis/_finetune/final_eval_v2"
MODELS = {"llama-3-8b": "29-32", "llama-3.1-8b": "29-32",
          "qwen1.5-7b": "29-32", "qwen2-7b": "25-28", "qwen2.5-7b": "25-28"}
rng = np.random.default_rng(0)


def acc(name):
    p = E / f"gsm8k_full__{name}.csv"
    return np.array([int(r["correct_stop"]) for r in csv.DictReader(open(p))]) if p.exists() else None


print("GSM8K, all training data, official test set (1,319), stop rule. ta_pull = separation loss, sign reversed,")
print("last 4 layers. Values in percentage points. SD = standard deviation across seeds.\n")
print(f"{'model':13s} {'n':>2s}  {'baseline mean±SD':>17s}  {'ta_pull mean±SD':>16s}  {'Δ mean±SD':>12s}  "
      f"{'better':>6s}  {'t-test p':>8s}  {'Wilcoxon p':>10s}  {'question-level 95% CI':>22s}")
for m in MODELS:
    pairs = []
    for s in range(10):
        b, u = acc(f"final_gsm8k_full_{m}_baseline_s{s}"), acc(f"final_gsm8k_full_{m}_ta_pull_s{s}")
        if b is not None and u is not None:
            pairs.append((b, u))
    if not pairs:
        print(f"{m:13s}  no results")
        continue
    ba = np.array([b.mean() for b, _ in pairs]) * 100
    ua = np.array([u.mean() for _, u in pairs]) * 100
    d = ua - ba
    n = len(d)
    t_p = stats.ttest_1samp(d, 0).pvalue if n > 1 else float("nan")
    w_p = stats.wilcoxon(d).pvalue if n > 1 and np.any(d != 0) else float("nan")
    diff = np.concatenate([u - b for b, u in pairs])
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() * 100 for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sd = lambda x: x.std(ddof=1) if len(x) > 1 else float("nan")
    print(f"{m:13s} {n:2d}  {ba.mean():7.1f} ± {sd(ba):4.1f}      {ua.mean():6.1f} ± {sd(ua):4.1f}     "
          f"{d.mean():+5.1f} ± {sd(d):4.1f}   {int((d > 0).sum()):2d}/{n:<2d}   {t_p:8.2f}  {w_p:10.2f}  "
          f"[{lo:+5.1f}, {hi:+5.1f}]")
    print(f"{'':13s}     per-seed Δ: {' '.join(f'{x:+.1f}' for x in d)}")
