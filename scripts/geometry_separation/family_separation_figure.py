"""Layer-wise separation of all 13 models averaged over the SemCor words, one panel per family.

    python scripts/geometry_separation/family_separation_figure.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = {
    "LLaMA Family": ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b"],
    "Qwen Family": ["qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"],
    "BERT-Family Encoders": ["bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base"],
}
FAMDIR = {"LLaMA Family": "llama", "Qwen Family": "qwen", "BERT-Family Encoders": "bert"}
WORDS = ["bank", "bat", "crane", "seal", "plant", "pupil", "club"]   # not SemCor, excluded
STYLE = {
    "llama-7b": ("#1f77b4", "o"), "llama-2-7b": ("#ff7f0e", "s"), "llama-3-8b": ("#2ca02c", "^"),
    "llama-3.1-8b": ("#d62728", "D"),
    "qwen-7b": ("#9467bd", "o"), "qwen1.5-7b": ("#8c564b", "s"), "qwen2-7b": ("#e377c2", "^"),
    "qwen2.5-7b": ("#17becf", "D"), "qwen3-8b": ("#bcbd22", "v"),
    "bert-base": ("#393b79", "o"), "roberta-base": ("#e6550d", "s"), "spanbert-base-cased": ("#31a354", "^"),
    "xlm-roberta-base": ("#7f7f7f", "D"),
}
OUT = ROOT / "results/analysis/_family_figure"


def sep(model, fam, word):
    p = ROOT / "results/words" / word / fam / model / "metrics/semantic_separation.csv"
    if not p.exists():
        print(f"  missing: {p}")
        return None
    return pd.read_csv(p).sort_values("layer").set_index("layer")["separation"]


def semcor_words():
    """SemCor words present for all 13 models (controlled words and pilot sets excluded)."""
    sets = []
    for name, models in FAMILIES.items():
        for m in models:
            ws = {p.parts[-5] for p in (ROOT / "results/words").glob(f"*/{FAMDIR[name]}/{m}/metrics/semantic_separation.csv")}
            sets.append(ws)
    common = set.intersection(*sets) - set(WORDS)
    common = {w for w in common if "_" not in w and "pilot" not in w and "minimal" not in w}
    print(f"SemCor words common to all 13 models: {len(common)}")
    return sorted(common)


SEMCOR = None


def curve(model, fam, which):
    global SEMCOR
    SEMCOR = SEMCOR or semcor_words()
    s = [sep(model, fam, w) for w in SEMCOR]
    return pd.concat([x for x in s if x is not None], axis=1).mean(axis=1)


def figure(which, title):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))
    rows = []
    for ax, (name, models) in zip(axes, FAMILIES.items()):
        for i, m in enumerate(models):
            c = curve(m, FAMDIR[name], which)
            if c is None:
                continue
            col, mk = STYLE[m]
            ax.plot(c.index, c.values, lw=2.4, color=col, marker=mk, markersize=5,
                    markevery=max(1, len(c) // 12), label=m)
            rows.append({"model": m, "peak Sep": round(c.max(), 4), "peak layer": int(c.idxmax()),
                         "layers": int(c.index.max())})
        ax.set_title(name, fontsize=16, weight="bold")
        ax.set_xlabel("Layer (0 = embedding output)", fontsize=13)
        ax.tick_params(labelsize=12)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.grid(axis="y", ls="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=11, frameon=True, framealpha=0.9, loc="best")
    axes[0].set_ylabel("Semantic separation Sep(l)", fontsize=13)
    fig.suptitle(title, fontsize=17, weight="bold")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"family_separation_{which}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")
    print(pd.DataFrame(rows).to_string(index=False), "\n")


figure("semcor", "Layer-wise semantic separation, mean over all SemCor words (target-token extraction)")