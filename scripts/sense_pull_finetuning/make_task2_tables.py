"""GSM8K and other-benchmark tables (png + csv) from the raw predictions in paper_final.

    cd paper_final; python code/scripts/sense_pull_finetuning/make_task2_tables.py"""
import argparse, csv, json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".", help="paper_final folder")
a = ap.parse_args()
PART = Path(a.root) / "04_sense_pull_finetuning"
GSM, BROAD, RUNS, OUT = PART / "data/gsm8k_predictions", PART / "data/benchmark_predictions", PART / "data/training_runs", PART / "tables"

N_LAYERS = {"llama-3-8b": 32, "llama-3.1-8b": 32, "qwen1.5-7b": 32, "qwen2-7b": 28, "qwen2.5-7b": 28}
VARIANT = {4: "ta_pull", 8: "ta_pull_last8", 10: "ta_pull_last10"}
SEEDS = (0, 1, 2)
BENCH = ["arc_c", "hellaswag", "piqa", "siqa", "csqa", "obqa", "mmlu", "mmlu_math", "strategyqa", "tydiqa", "theoremqa"]
BNAME = ["ARC-C", "HellaSwag", "PIQA", "SIQA", "CSQA", "OBQA", "MMLU", "MMLU\nmath", "StrategyQA", "TyDiQA", "TheoremQA"]


def mean_col(p, col):
    return float(np.mean([int(r[col]) for r in csv.DictReader(open(p))])) * 100


def gsm(m, v, s):
    return mean_col(GSM / f"gsm8k_full__final_gsm8k_full_{m}_{v}_s{s}.csv", "correct_stop")


def broad(m, v, s):
    return [mean_col(BROAD / f"{b}__final_gsm8k_full_{m}_{v}_s{s}.csv", "correct") for b in BENCH]


def cos_sep(m, v, s):
    return json.load(open(RUNS / f"final_gsm8k_full_{m}_{v}_s{s}.json"))["geo_after"]["cos_sep"]


sd = lambda x: float(np.std(x, ddof=1))
fmt = lambda x, d=1: f"{x:+.{d}f}".replace("-", "−")


def table_png(rows, cols, widths, title, note, path, figsize, hi_col, color_cols=()):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center", colWidths=widths)
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 1.75)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if r == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(weight="bold")
            cell.set_height(cell.get_height() * 1.4)
            continue
        if (r - 1) // 3 % 2:
            cell.set_facecolor("#f7f7f7")
        if c in color_cols:
            v = float(cell.get_text().get_text().replace("−", "-"))
            if v <= -1.0:
                cell.get_text().set_color("#b03030")
            elif v >= 1.0:
                cell.get_text().set_color("#2a7a2a")
        if c == hi_col:
            cell.set_facecolor("#fff4d6")
            cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=12.5, weight="bold", loc="left")
    fig.text(0.125, 0.065, note, fontsize=9, va="top", color="#444")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# GSM8K
g_rows, g_csv = [], []
for m, n in N_LAYERS.items():
    base = np.array([gsm(m, "baseline", s) for s in SEEDS])
    for i, k in enumerate((4, 8, 10)):
        acc = np.array([gsm(m, VARIANT[k], s) for s in SEEDS])
        d = acc - base
        tp = stats.ttest_1samp(d, 0).pvalue
        wp = stats.wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
        cs = np.mean([cos_sep(m, VARIANT[k], s) for s in SEEDS])
        lay = f"{n - k + 1}–{n}"
        g_rows.append([m if i == 0 else "", f"{base.mean():.1f} ± {sd(base):.1f}" if i == 0 else "", lay,
                       f"{acc.mean():.1f} ± {sd(acc):.1f}", f"{fmt(d.mean())} ± {sd(d):.1f}", f"{int((d > 0).sum())}/3",
                       f"{tp:.2f} / {wp:.2f}", " ".join(fmt(x) for x in d), f"{cs:.3f}"])
        g_csv.append([m, f"{n - k + 1}-{n}", round(base.mean(), 2), round(sd(base), 2), round(acc.mean(), 2), round(sd(acc), 2),
                      round(d.mean(), 3), round(sd(d), 3), int((d > 0).sum()), round(tp, 3), round(wp, 3), *np.round(d, 2), round(cs, 4)])

seeds10 = [s for s in range(10) if (GSM / f"gsm8k_full__final_gsm8k_full_llama-3-8b_ta_pull_s{s}.csv").exists()]
d10 = np.array([gsm("llama-3-8b", "ta_pull", s) - gsm("llama-3-8b", "baseline", s) for s in seeds10])
note10 = (f"llama-3-8b, last 4 layers with {len(seeds10)} seeds: Δ = {fmt(d10.mean())} ± {sd(d10):.1f} points, "
          f"{int((d10 > 0).sum())}/{len(seeds10)} seeds better, t-test p = {stats.ttest_1samp(d10, 0).pvalue:.2f}.")
table_png(g_rows,
          ["Model", "Baseline\nmean ± SD (%)", "Regularised\nlayers", "Pull\nmean ± SD (%)", "Δ pull − baseline\nmean ± SD (points)",
           "Seeds\nbetter", "Across-seed test\nt-test p / Wilcoxon p", "Δ per seed\n(s0  s1  s2)", "cos_sep after\n(regularised layers)"],
          [0.09, 0.1, 0.08, 0.1, 0.13, 0.06, 0.13, 0.13, 0.1],
          "GSM8K: pulling the two senses together on the last 4 / 8 / 10 layers — all training data, official test set, 3 seeds",
          "Generation accuracy on the official GSM8K test set (1,319 questions). Same baseline (seeds 0–2) for the three layer settings of a model. "
          "SD = standard deviation across seeds.\nAcross-seed test: one-sample t-test and Wilcoxon signed-rank test of the 3 per-seed differences "
          "(with 3 seeds the smallest possible Wilcoxon p is 0.25).\n" + note10 + "\n"
          "cos_sep = same-sense − cross-sense cosine on 50 held-out SemCor words, on the regularised layers. LoRA r = 16, λ = 1, 1,991 steps.",
          OUT / "gsm8k_pull_layers.png", (17, 6.2), hi_col=4)
with open(OUT / "gsm8k_pull_layers.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "layers", "baseline_mean", "baseline_sd", "pull_mean", "pull_sd", "delta_mean", "delta_sd",
                "seeds_better", "t_p", "wilcoxon_p", "delta_s0", "delta_s1", "delta_s2", "cos_sep_after"])
    w.writerows(g_csv)

# other benchmarks
b_rows, b_csv = [], []
for m, n in N_LAYERS.items():
    base = np.array([broad(m, "baseline", s) for s in SEEDS])
    for i, k in enumerate((4, 8, 10)):
        d = np.array([broad(m, VARIANT[k], s) for s in SEEDS]) - base
        mean10 = d[:, :10].mean(1)
        p = stats.ttest_1samp(mean10, 0).pvalue
        b_rows.append([m if i == 0 else "", f"{n - k + 1}–{n}", f"{base[:, :10].mean():.1f}" if i == 0 else ""]
                      + [fmt(x) for x in d.mean(0)]
                      + [f"{fmt(mean10.mean())} ± {sd(mean10):.1f}", f"{int((mean10 > 0).sum())}/3", f"{p:.2f}"])
        b_csv.append([m, f"{n - k + 1}-{n}", round(base[:, :10].mean(), 2), *np.round(d.mean(0), 2),
                      round(mean10.mean(), 2), round(sd(mean10), 2), int((mean10 > 0).sum()), round(p, 3)])
table_png(b_rows, ["Model", "Layers", "Baseline\nmean (%)"] + BNAME + ["Δ mean (10)\nmean ± SD", "Seeds\nbetter", "t-test\np"],
          [0.075, 0.05, 0.055] + [0.052] * 11 + [0.085, 0.045, 0.045],
          "Other benchmarks: pulling the two senses together on the last 4 / 8 / 10 layers vs the GSM8K baseline — Δ in points, 3 seeds",
          "Models fine-tuned on all GSM8K training data (7,473). Each value = pull − same-seed baseline, mean of seeds 0–2. "
          "Baseline mean = baseline accuracy over the 10 benchmarks (TheoremQA not in the mean).\n"
          "Δ mean (10) = per-seed mean Δ over the 10 benchmarks, then mean ± SD across the 3 seeds; "
          "t-test = one-sample t-test of the 3 per-seed values. Red ≤ −1.0, green ≥ +1.0.\n"
          "Multiple choice scored by mean log-probability per token; TyDiQA and TheoremQA by generation. LoRA r = 16, λ = 1, 1,991 steps.",
          OUT / "broad_pull_layers.png", (20, 6.2), hi_col=14, color_cols=range(3, 14))
with open(OUT / "broad_pull_layers.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "layers", "baseline_mean10"] + BENCH + ["delta_mean10", "delta_mean10_sd", "seeds_better", "t_p"])
    w.writerows(b_csv)

print(f"written to {OUT}: gsm8k_pull_layers.png/.csv, broad_pull_layers.png/.csv")
for r in g_rows:
    print(f"  GSM8K  {r[0] or '':13s} {r[2]:>6s}  Δ {r[4]}")
