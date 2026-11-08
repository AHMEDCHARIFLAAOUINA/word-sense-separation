"""Compare the project's measure (1 - cos of the sense-average vectors) with our Sep(l).

    cd ta_sememe_project; python our_scripts/compare_ta_ours.py"""
import os
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
PIPE = Path(os.environ.get("WSS_PIPELINE", "."))
OUT = ROOT / "outputs" / "comparison"
FAMILIES = {
    "LLaMA Family": ("llama", ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b"]),
    "Qwen Family": ("qwen", ["qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"]),
    "BERT-Family Encoders": ("bert", ["bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base"]),
}
STYLE = {
    "llama-7b": ("#1f77b4", "o"), "llama-2-7b": ("#ff7f0e", "s"), "llama-3-8b": ("#2ca02c", "^"),
    "llama-3.1-8b": ("#d62728", "D"), "qwen-7b": ("#9467bd", "o"), "qwen1.5-7b": ("#8c564b", "s"), "qwen2-7b": ("#e377c2", "^"),
    "qwen2.5-7b": ("#17becf", "D"), "qwen3-8b": ("#bcbd22", "v"), "bert-base": ("#393b79", "o"),
    "roberta-base": ("#e6550d", "s"), "spanbert-base-cased": ("#31a354", "^"), "xlm-roberta-base": ("#7f7f7f", "D"),
}
CONTROLLED = ["bank", "bat", "crane", "seal", "plant", "pupil", "club"]


def dataset_words(ds):
    if ds == "controlled7":
        return CONTROLLED
    return sorted(p.stem for p in (PIPE / "data/raw/semcor_words").glob("*.json") if not p.name.startswith("_"))


def ours(ds, fam, model):
    curves = []
    for w in dataset_words(ds):
        p = PIPE / "results/words" / w / fam / model / "metrics/semantic_separation.csv"
        if p.exists():
            curves.append(pd.read_csv(p).sort_values("layer").set_index("layer")["separation"])
    return (pd.concat(curves, axis=1).mean(axis=1), len(curves)) if curves else (None, 0)


def project(ds, model):
    p = ROOT / "outputs" / ds / "analysis" / model / "mean_cosine_by_layer.csv"
    if not p.exists():
        return None, 0
    d = pd.read_csv(p, encoding="utf-8-sig").sort_values("layer")
    s = 1 - d.set_index("layer")["mean_cosine_similarity"].astype(float)
    return s, int(d["number_of_word_sememe_pairs"].max())


OUT.mkdir(parents=True, exist_ok=True)
for ds, title in (("controlled7", "7 controlled words"), ("semcor", "SemCor words")):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    rows = []
    for c, (fname, (fam, models)) in enumerate(FAMILIES.items()):
        for m in models:
            o, n_o = ours(ds, fam, m)
            h, n_h = project(ds, m)
            col, mk = STYLE[m]
            kw = dict(color=col, marker=mk, markersize=4, lw=2, label=m)
            if o is not None:
                axes[0, c].plot(o.index, o.values, markevery=max(1, len(o) // 12), **kw)
            if h is not None:
                axes[1, c].plot(h.index, h.values, markevery=max(1, len(h) // 12), **kw)
            row = {"model": m, "words (ours)": n_o, "words (project)": n_h}
            if o is not None and h is not None and len(o) == len(h):
                a, b = o.values[1:], h.values[1:]   # layers 1..L (block outputs)
                row.update({"peak layer (ours)": int(o.iloc[1:].idxmax()), "peak layer (project)": int(h.iloc[1:].idxmax()),
                            "Pearson r": round(pearsonr(a, b)[0], 3), "Spearman rho": round(spearmanr(a, b)[0], 3)})
            if h is None:
                row.update({"note": "the project code cannot run this model (no fast tokenizer)"})
            rows.append(row)
        axes[0, c].set_title(fname, fontsize=15, weight="bold")
        for r in (0, 1):
            ax = axes[r, c]
            ax.axhline(0, color="grey", lw=0.8, ls="--")
            ax.grid(axis="y", ls="--", alpha=0.35)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=10, frameon=True, framealpha=0.9)
            ax.set_xlabel("Layer (0 = embedding output)")
    axes[0, 0].set_ylabel("OURS: Sep(l)\nsame-sense cos − different-sense cos", fontsize=12)
    axes[1, 0].set_ylabel("OpenHowNet project code: 1 − cos(sense-average vectors)", fontsize=12)
    axes[1, 1].text(0.02, 0.97, "qwen-7b: not run with the project code (no fast tokenizer)", va="top",
                    transform=axes[1, 1].transAxes, fontsize=10, color="#555")
    fig.suptitle(f"Same data, two methods — {title} (human sense labels); higher = senses more separated",
                 fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / f"compare_{ds}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    t = pd.DataFrame(rows)
    for c in ("peak layer (ours)", "peak layer (project)"):
        if c in t:
            t[c] = t[c].astype("Int64")
    if "note" in t:
        t["note"] = t["note"].fillna("")
    t.to_csv(OUT / f"compare_{ds}.csv", index=False)
    print(f"=== {title}")
    print(t.to_string(index=False), "\n")
print(f"figures and tables -> {OUT}")
