"""Dataset table (Table 1) and the distribution figures.

    python scripts/geometry_separation/semcor_table1_and_figures.py"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/analysis/_semcor_final"
DEC = ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b", "qwen-7b",
       "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"]
ENC = ["bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base"]


def jaccard(words):
    rows = []
    for w in words:
        d = json.load(open(f"data/raw/semcor_words/{w}.json"))
        by = {}
        for s, lab in zip(d["sentences"], d["labels"]):
            by.setdefault(lab, set()).update(t.lower().strip(".,;:!?\"'()")
                                             for t in s.split())
        a, b = list(by.values())
        rows.append(dict(word=w, jaccard=len(a & b) / max(1, len(a | b)),
                         length=np.mean([len(s.split()) - 2 for s in d["sentences"]]),
                         n_sentences=len(d["sentences"])))
    return pd.DataFrame(rows)


def sep_stats():
    """Sep max and its relative depth, per word, averaged over the decoders."""
    rows = []
    for f in glob.glob("results/words/*/*/*/metrics/*separation*.csv"):
        parts = f.split("/")
        word, model = parts[2], parts[4]
        if model not in DEC:
            continue
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        col = next((c for c in d.columns if "sep" in c.lower()
                    and d[c].dtype != object), None)
        if col is None or d.empty:
            continue
        v = d[col].values
        rows.append(dict(word=word, model=model, sep_max=np.nanmax(v),
                         peak_depth=int(np.nanargmax(v)) / max(1, len(v) - 1)))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    man = json.load(open("data/raw/semcor_words/_manifest.json"))["words"]
    words = [w["word"] for w in man]

    # ---------------------------------------------------------------- Table 1
    T = jaccard(words)
    S = sep_stats()
    if not S.empty:
        S = S[S.word.isin(words)].groupby("word")[["sep_max", "peak_depth"]].mean()
        T = T.merge(S, on="word", how="left")

    cb = pd.read_csv("results/analysis/_context_baseline/context_baseline_semcor.csv")
    piv = cb.pivot_table(index="word", columns="representation", values="accuracy")
    T = T.merge(piv.reset_index(), on="word", how="left")
    T.to_csv(f"{OUT}/table1_semcor_per_word.csv", index=False)

    def q(c, p):
        return T[c].quantile(p) if c in T else np.nan
    summary = pd.DataFrame([dict(
        dataset="SemCor 1,000 words", n_words=len(T),
        n_sentences=int(T.n_sentences.sum()),
        jaccard_median=T.jaccard.median(), length_median=T.length.median(),
        context_median=q("context", .5), target_median=q("target", .5),
        sep_max_median=q("sep_max", .5), peak_depth_median=q("peak_depth", .5),
        context_p90=q("context", .9), share_context_ge_09=(T.get("context") >= .9).mean())])
    summary.to_csv(f"{OUT}/table1_semcor_summary.csv", index=False)

    print("TABLE 1  (medians over the 1,000 words; decoder means per word)")
    for k, v in summary.iloc[0].items():
        print(f"  {k:24s} {v if isinstance(v, str) else round(float(v), 4)}")

    # ---------------------------------------------------------------- figures
    df = pd.read_csv("results/analysis/_cross_word/cross_word_q_proj_stable_rank_semcor.csv")
    rob = pd.read_csv("results/analysis/_cross_word/"
                      "cross_word_q_proj_stable_rank_semcor_robustness_models.csv")
    order = [m for m in DEC + ENC if m in set(df.model)]

    # 1. distribution of per-word r
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [df[df.model == m].partial_r.dropna().values for m in order]
    bp = ax.boxplot(data, vert=False, labels=order, showfliers=False,
                    patch_artist=True, widths=.6)
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor("#7fb2d6" if m in DEC else "#e08b8b")
        patch.set_edgecolor("black")
    ax.axvline(0, color="grey", lw=1, ls="--")
    ax.set_xlabel("partial $r$ per word (linear depth control)")
    ax.set_title("q_proj stable rank against Sep($l$): distribution over 1,000 words")
    fig.tight_layout(); fig.savefig(f"{OUT}/figS_semcor1_distribution.pdf")
    fig.savefig(f"{OUT}/figS_semcor1_distribution.png", dpi=150); plt.close(fig)

    # 2. per-model mean with bootstrap interval
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, mark, off in (("linear", "o", -.15), ("quadratic", "s", .15)):
        r = rob[rob.control == label].set_index("model").reindex(order)
        y = np.arange(len(order)) + off
        ax.errorbar(r["mean"], y, xerr=[r["mean"] - r.ci_lo, r.ci_hi - r["mean"]],
                    fmt=mark, ms=5, capsize=3, lw=1,
                    label=f"{label} depth control",
                    color="black" if label == "linear" else "none",
                    markerfacecolor="black" if label == "linear" else "white",
                    markeredgecolor="black", ecolor="grey")
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels(order)
    ax.axvline(0, color="grey", lw=1, ls="--")
    ax.axhline(len(DEC) - .5, color="black", lw=.8)
    ax.invert_yaxis(); ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("mean partial $r$ over words (bars: bootstrap 95% interval)")
    fig.tight_layout(); fig.savefig(f"{OUT}/figS_semcor2_model_means.pdf")
    fig.savefig(f"{OUT}/figS_semcor2_model_means.png", dpi=150); plt.close(fig)

    # 3. context-only against target-only
    if {"context", "target"} <= set(piv.columns):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(piv["context"], piv["target"], s=8, alpha=.35,
                   color="#3b6ea5", edgecolors="none")
        ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
        ax.axvline(.5, color="grey", lw=.8, ls=":")
        ax.set_xlabel("context-only accuracy (target word deleted)")
        ax.set_ylabel("target-only accuracy")
        ax.set_title("Sense recoverability per word")
        fig.tight_layout(); fig.savefig(f"{OUT}/figS_semcor3_context_baseline.pdf")
        fig.savefig(f"{OUT}/figS_semcor3_context_baseline.png", dpi=150); plt.close(fig)

    print("\nwritten to", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)


if __name__ == "__main__":
    main()
