"""Other-benchmark results: delta vs the same-seed baseline per benchmark and the 10-benchmark mean.

    python scripts/sense_pull_finetuning/broad_full_summary.py"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "results/analysis/_finetune/broad_eval"
MODELS = ("llama-3-8b", "llama-3.1-8b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b")
BENCH = ["arc_c", "hellaswag", "piqa", "siqa", "csqa", "obqa", "mmlu", "mmlu_math", "strategyqa", "tydiqa"]
SHORT = ["ARC-C", "HellaS", "PIQA", "SIQA", "CSQA", "OBQA", "MMLU", "MMLU-m", "StratQA", "TyDiQA"]
SEEDS = (0, 1, 2)


def acc(b, name):
    p = E / f"{b}__{name}.csv"
    return float(np.mean([int(r["correct"]) for r in csv.DictReader(open(p))])) * 100 if p.exists() else None


print("Broad evaluation of the GSM8K-full models (all GSM8K training data). Baseline / pretrained rows: accuracy %;")
print("ta_pull rows: Δ vs same-seed baseline in points, mean of 3 seeds. Mean = 10 benchmarks (not TheoremQA);")
print("for ta_pull rows the mean column also shows ± SD across seeds, seeds better and the across-seed t-test p.\n")
for m in MODELS:
    print(f"=== {m}")
    print(f"  {'':16s}" + "".join(f"{s:>8s}" for s in SHORT) + f"{'Theorem':>9s}{'Mean':>10s}")
    pre = "pretrained" if m == "llama-3-8b" else f"pretrained_{m}"
    row = [acc(b, pre) for b in BENCH + ["theoremqa"]]
    if all(x is not None for x in row[:-1]):
        print(f"  {'pretrained':16s}" + "".join(f"{x:8.1f}" for x in row[:-1])
              + (f"{row[-1]:9.1f}" if row[-1] is not None else f"{'-':>9s}") + f"{np.mean(row[:-1]):10.1f}")
    base = {s: [acc(b, f"final_gsm8k_full_{m}_baseline_s{s}") for b in BENCH + ["theoremqa"]] for s in SEEDS}
    if any(x is None for s in SEEDS for x in base[s][:-1]):
        print("  baseline missing\n")
        continue
    bm = np.array([base[s] for s in SEEDS], dtype=float)
    print(f"  {'baseline':16s}" + "".join(f"{x:8.1f}" for x in bm.mean(0)[:-1]) + f"{bm.mean(0)[-1]:9.1f}"
          + f"{bm[:, :-1].mean():10.1f}")
    for v, label in (("ta_pull", "pull last 4"), ("ta_pull_last8", "pull last 8"), ("ta_pull_last10", "pull last 10")):
        r = {s: [acc(b, f"final_gsm8k_full_{m}_{v}_s{s}") for b in BENCH + ["theoremqa"]] for s in SEEDS}
        if any(x is None for s in SEEDS for x in r[s][:-1]):
            print(f"  {label:16s} missing")
            continue
        d = np.array([r[s] for s in SEEDS], dtype=float) - bm
        mean10 = d[:, :-1].mean(1)   # per-seed Δ of the 10-benchmark mean
        p = stats.ttest_1samp(mean10, 0).pvalue
        print(f"  {label:16s}" + "".join(f"{x:+8.1f}" for x in d.mean(0)[:-1]) + f"{d.mean(0)[-1]:+9.1f}"
              + f"{mean10.mean():+7.1f} ± {mean10.std(ddof=1):.1f} ({int((mean10 > 0).sum())}/3, p {p:.2f})"
              + f"   per seed {' '.join(f'{x:+.1f}' for x in mean10)}")
    print()
