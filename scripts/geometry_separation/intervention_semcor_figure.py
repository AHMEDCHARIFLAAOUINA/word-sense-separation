"""Figures of the spectral intervention.

    python scripts/geometry_separation/intervention_semcor_figure.py"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/analysis/_semcor_final"
DEC = ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b", "qwen-7b",
       "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"]


def boot(v, n=10000, seed=0):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    t = pd.read_csv(f"{OUT}/intervention_semcor_per_word.csv")
    t = t[t.peak_ret.notna()]

    # ---- figS_semcor6: distribution of peak retention per model
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, proj in zip(axes, ["q_proj", "v_proj"]):
        data, labels = [], []
        for m in DEC:
            v = t[(t.model == m) & (t.projection == proj)].peak_ret
            if len(v):
                data.append(v.values); labels.append(m)
        bp = ax.boxplot(data, vert=False, tick_labels=labels, showfliers=False,
                        patch_artist=True, widths=.6)
        for patch in bp["boxes"]:
            patch.set_facecolor("#7fb2d6" if proj == "q_proj" else "#e0a27f")
            patch.set_edgecolor("black")
        ax.axvline(1.0, color="grey", lw=1, ls="--")
        ax.set_xlabel("peak Sep($l$) retention")
        ax.set_title(f"{proj} truncated to 80% of stable rank", fontsize=11)
        ax.invert_yaxis()
    axes[0].set_xlim(0.7, 1.35)
    fig.suptitle("Spectral intervention over 1,000 words "
                 "(retention against each word's own baseline)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figS_semcor6_intervention_distribution.pdf")
    fig.savefig(f"{OUT}/figS_semcor6_intervention_distribution.png", dpi=150)
    plt.close(fig)

    # ---- figS_semcor7: per-model mean change with bootstrap interval
    fig, ax = plt.subplots(figsize=(8, 5))
    for proj, mark, off, fc in (("q_proj", "o", -.15, "black"),
                                ("v_proj", "s", .15, "white")):
        means, los, his = [], [], []
        for m in DEC:
            v = t[(t.model == m) & (t.projection == proj)].peak_ret.dropna()
            lo, hi = boot(v)
            means.append(v.mean() - 1); los.append(lo - 1); his.append(hi - 1)
        y = np.arange(len(DEC)) + off
        means = np.array(means); los = np.array(los); his = np.array(his)
        ax.errorbar(means, y, xerr=[means - los, his - means], fmt=mark, ms=5,
                    capsize=3, lw=1, color="black", markerfacecolor=fc,
                    markeredgecolor="black", ecolor="grey", label=proj)
    ax.set_yticks(np.arange(len(DEC))); ax.set_yticklabels(DEC)
    ax.axvline(0, color="grey", lw=1, ls="--")
    ax.invert_yaxis(); ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("mean peak retention $-$ 1  (bars: bootstrap 95% interval)")
    ax.set_title("Change in peak separation after truncation, 1,000 words",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figS_semcor7_intervention_model_means.pdf")
    fig.savefig(f"{OUT}/figS_semcor7_intervention_model_means.png", dpi=150)
    plt.close(fig)

    print("written:")
    for f in sorted(os.listdir(OUT)):
        if "semcor6" in f or "semcor7" in f:
            print("  ", f)


if __name__ == "__main__":
    main()
