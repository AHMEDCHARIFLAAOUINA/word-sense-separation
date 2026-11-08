"""Separation by architecture and the measure-comparison figure.

    python scripts/geometry_separation/semcor_layer_figures.py"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/analysis/_semcor_final"
FAM = {"LLaMA family": ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b"],
       "Qwen family": ["qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"],
       "BERT-family encoders": ["bert-base", "roberta-base",
                                "spanbert-base-cased", "xlm-roberta-base"]}
GRID = np.linspace(0, 1, 41)   # common relative-depth grid


def curves(model, words):
    """Sep(l) resampled onto the relative-depth grid, one row per word."""
    rows = []
    for w in words:
        hits = glob.glob(f"results/words/{w}/*/{model}/metrics/*separation*.csv")
        if not hits:
            continue
        d = pd.read_csv(hits[0])
        if "separation" not in d or len(d) < 4:
            continue
        v = d["separation"].values.astype(float)
        depth = np.linspace(0, 1, len(v))
        rows.append(np.interp(GRID, depth, v))
    return np.array(rows) if rows else None


def main():
    os.makedirs(OUT, exist_ok=True)
    words = [w["word"] for w in
             json.load(open("data/raw/semcor_words/_manifest.json"))["words"]]

    # ------------------------------------------------ figS_semcor4: Sep(l)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    summary = []
    for ax, (fam, models) in zip(axes, FAM.items()):
        stack = []
        for m in models:
            C = curves(m, words)
            if C is None:
                continue
            stack.append(C)
            med = np.nanmedian(C, axis=0)
            ax.plot(GRID, med, lw=1, alpha=.55, label=m)
        if not stack:
            continue
        allc = np.vstack(stack)
        med = np.nanmedian(allc, axis=0)
        lo = np.nanpercentile(allc, 25, axis=0)
        hi = np.nanpercentile(allc, 75, axis=0)
        ax.fill_between(GRID, lo, hi, color="grey", alpha=.18, lw=0)
        ax.plot(GRID, med, color="black", lw=2, label="family median")
        k = int(np.nanargmax(med))
        ax.plot(GRID[k], med[k], "ks", ms=5)
        ax.annotate(f"peak {GRID[k]:.2f}", (GRID[k], med[k]),
                    textcoords="offset points", xytext=(4, -12), fontsize=8)
        ax.set_title(fam, fontsize=11)
        ax.set_xlabel("Relative depth")
        ax.legend(fontsize=7, frameon=False)
        summary.append(dict(family=fam, n_models=len(stack), n_curves=len(allc),
                            sep_max_median=float(med[k]), peak_depth=float(GRID[k])))
    axes[0].set_ylabel("Semantic separation  Sep($l$)")
    fig.suptitle("Layer-wise semantic separation over 1,000 SemCor words "
                 "(median per word; band: interquartile range)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figS_semcor4_separation_by_architecture.pdf")
    fig.savefig(f"{OUT}/figS_semcor4_separation_by_architecture.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(summary).to_csv(f"{OUT}/figS_semcor4_family_peaks.csv", index=False)
    print("family peaks")
    for r in summary:
        print(f"  {r['family']:24s} curves {r['n_curves']:5d}  "
              f"Sep max {r['sep_max_median']:.4f}  peak depth {r['peak_depth']:.2f}")

    # ------------------------------ figS_semcor5: primary against comparison
    q = pd.read_csv("results/analysis/_cross_word/"
                    "cross_word_q_proj_stable_rank_semcor_models.csv")
    u = pd.read_csv("results/analysis/_cross_word/"
                    "cross_word_up_proj_erank_semcor_models.csv")
    order = [m for fam in FAM.values() for m in fam if m in set(q.model)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (col, label) in zip(axes, (("mean_r", "linear depth control"),
                                       ("mean_r_quad", "quadratic depth control"))):
        qi = q.set_index("model").reindex(order)
        ui = u.set_index("model").reindex(order)
        y = np.arange(len(order))
        for i in y:
            ax.plot([ui[col].iloc[i], qi[col].iloc[i]], [i, i],
                    color="grey", lw=.8, zorder=1)
        ax.scatter(qi[col], y, s=28, color="black", zorder=2,
                   label="q_proj stable rank")
        ax.scatter(ui[col], y, s=28, facecolors="white", edgecolors="black",
                   zorder=2, label="up_proj effective rank")
        ax.axvline(0, color="grey", lw=1, ls="--")
        ax.axhline(3.5, color="black", lw=.8)
        ax.axhline(8.5, color="black", lw=.8)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("mean partial $r$ over 1,000 words")
    axes[0].set_yticks(np.arange(len(order))); axes[0].set_yticklabels(order)
    axes[0].invert_yaxis(); axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Primary against comparison measure, 1,000 words", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figS_semcor5_measure_comparison.pdf")
    fig.savefig(f"{OUT}/figS_semcor5_measure_comparison.png", dpi=150)
    plt.close(fig)

    print("\nwritten to", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)


if __name__ == "__main__":
    main()
