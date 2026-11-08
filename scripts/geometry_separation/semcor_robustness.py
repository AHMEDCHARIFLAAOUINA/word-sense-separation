"""Bootstrap over words for the per-model and group estimates.

    python scripts/geometry_separation/semcor_robustness.py --measure stable_rank --projection q_proj --tag semcor"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy import stats

DEC = ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b", "qwen-7b",
       "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"]
ENC = ["bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base"]
LLAMA = [m for m in DEC if m.startswith("llama")]
QWEN = [m for m in DEC if m.startswith("qwen")]


def boot(values, n=10000, seed=0):
    """Percentile bootstrap of the mean."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan, np.nan, np.nan
    m = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", default="stable_rank")
    ap.add_argument("--projection", default="q_proj")
    ap.add_argument("--tag", default="semcor")
    ap.add_argument("--manifest", default="data/raw/semcor_words/_manifest.json")
    ap.add_argument("--min-words", type=int, default=900,
                    help="models with fewer analysed words are reported but flagged")
    a = ap.parse_args()

    base = f"results/analysis/_cross_word/cross_word_{a.projection}_{a.measure}"
    if a.tag:
        base += f"_{a.tag}"
    df = pd.read_csv(f"{base}.csv")

    n_sent = {w["word"]: 2 * w["n_per_sense"]
              for w in json.load(open(a.manifest))["words"]}
    df["n_sent"] = df["word"].map(n_sent)

    rcol = "partial_r" if "partial_r" in df.columns else df.columns[df.columns.str.fullmatch(r"r|partial_r|r_linear")][0]
    qcol = [c for c in df.columns if "quad" in c and df[c].dtype != object][0]
    print(f"using columns: linear={rcol}, quadratic={qcol}\n")

    rows = []
    for model, g in df.groupby("model"):
        for label, col in (("linear", rcol), ("quadratic", qcol)):
            v = g[col].dropna()
            m, lo, hi = boot(v)
            w = np.average(g[col].dropna(), weights=g.loc[g[col].notna(), "n_sent"])
            pos = int((v > 0).sum())
            p = stats.binomtest(pos, len(v), 0.5).pvalue if len(v) else np.nan
            rows.append(dict(model=model, control=label, n_words=len(v),
                             mean=m, ci_lo=lo, ci_hi=hi, weighted_mean=w,
                             pos=pos, share_pos=pos / len(v) if len(v) else np.nan,
                             binom_p=p))
    M = pd.DataFrame(rows)

    print("PER MODEL  (bootstrap over words, 10,000 resamples)")
    print(f"  {'model':22s} {'control':10s} {'words':>6s} {'mean':>8s} "
          f"{'95% CI':>18s} {'weighted':>9s} {'pos':>7s} {'binom p':>9s}")
    for _, r in M.iterrows():
        flag = "" if r.n_words >= a.min_words else "  << incomplete"
        print(f"  {r.model:22s} {r.control:10s} {r.n_words:6d} {r['mean']:+8.3f} "
              f"[{r.ci_lo:+6.3f},{r.ci_hi:+6.3f}] {r.weighted_mean:+9.3f} "
              f"{100*r.share_pos:6.1f}% {r.binom_p:9.2g}{flag}")

    print("\nGROUP LEVEL  (one value per model, as in the main analysis)")
    groups = {"decoders": DEC, "llama": LLAMA, "qwen": QWEN, "encoders": ENC}
    out = []
    for gname, members in groups.items():
        for label in ("linear", "quadratic"):
            sub = M[(M.control == label) & (M.model.isin(members)) &
                    (M.n_words >= a.min_words)]
            if sub.empty:
                continue
            v = sub["mean"].values
            t_p = stats.ttest_1samp(v, 0).pvalue if len(v) > 1 else np.nan
            w_p = stats.wilcoxon(v).pvalue if len(v) > 1 else np.nan
            m, lo, hi = boot(v, seed=1)
            print(f"  {gname:10s} {label:10s} n={len(v)}  mean {v.mean():+.3f}  "
                  f"positive {int((v > 0).sum())}/{len(v)}  "
                  f"t p {t_p:.4f}  Wilcoxon p {w_p:.4f}")
            out.append(dict(group=gname, control=label, n_models=len(v),
                            mean=v.mean(), n_positive=int((v > 0).sum()),
                            t_p=t_p, wilcoxon_p=w_p, boot_lo=lo, boot_hi=hi))

    os.makedirs("results/analysis/_cross_word", exist_ok=True)
    M.to_csv(f"{base}_robustness_models.csv", index=False)
    pd.DataFrame(out).to_csv(f"{base}_robustness_groups.csv", index=False)
    print(f"\nwritten: {base}_robustness_models.csv, _robustness_groups.csv")


if __name__ == "__main__":
    main()
