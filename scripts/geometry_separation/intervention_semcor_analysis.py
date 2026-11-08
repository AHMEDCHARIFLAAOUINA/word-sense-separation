"""Model-level results of the spectral intervention.

    python scripts/geometry_separation/intervention_semcor_analysis.py"""
import glob, os
import numpy as np
import pandas as pd
from scipy import stats

BASE = "results/analysis/_intervention_semcor"
OUT = "results/analysis/_semcor_final"
FILES = ["intervention_semcor_q_and_v.csv", "intervention_semcor_rest.csv",
         "intervention_semcor_qwen7b.csv"]
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
    os.makedirs(OUT, exist_ok=True)
    d = pd.concat([pd.read_csv(f"{BASE}/{f}") for f in FILES
                   if os.path.exists(f"{BASE}/{f}")], ignore_index=True)
    d = d.drop_duplicates(subset=["model", "word", "projection", "ratio"],
                          keep="last")

    # retention against the same model/word/projection at ratio 1.0
    ref = (d[d.ratio == 1.0]
           .set_index(["model", "word", "projection"])[["sep_max", "sep_mean"]])
    t = d[d.ratio < 1.0].copy()
    key = list(zip(t.model, t.word, t.projection))
    t["sep_max_ref"] = [ref.sep_max.get(k, np.nan) for k in key]
    t["sep_mean_ref"] = [ref.sep_mean.get(k, np.nan) for k in key]
    t["peak_ret"] = t.sep_max / t.sep_max_ref
    t["mean_ret"] = t.sep_mean / t.sep_mean_ref
    t = t.replace([np.inf, -np.inf], np.nan)
    # drop words whose baseline Sep max is below 0.01: the ratio is meaningless there
    # (this is why fewer than 1,000 words per model enter the intervention tables)
    MIN_SEP = 0.01
    weak = (t.sep_max_ref.abs() < MIN_SEP)
    print(f"dropped {int(weak.sum())} of {len(t)} cells with baseline Sep max "
          f"below {MIN_SEP}")
    t.loc[weak, ["peak_ret", "mean_ret"]] = np.nan
    t.to_csv(f"{OUT}/intervention_semcor_per_word.csv", index=False)

    # ------------------------------------------------------- per model (5b)
    rows = []
    for (proj, model), g in t.groupby(["projection", "model"]):
        pk, mn = g.peak_ret.dropna(), g.mean_ret.dropna()
        lo, hi = boot(pk)
        rows.append(dict(projection=proj, model=model, n_words=len(pk),
                         peak_median=pk.median(), peak_mean=pk.mean(),
                         peak_ci_lo=lo, peak_ci_hi=hi,
                         peak_share_up=(pk > 1).mean(),
                         mean_median=mn.median(), mean_mean=mn.mean(),
                         mean_share_up=(mn > 1).mean(),
                         peak_p10=pk.quantile(.10), peak_p90=pk.quantile(.90)))
    M = pd.DataFrame(rows).sort_values(["projection", "model"])
    M.to_csv(f"{OUT}/intervention_semcor_models.csv", index=False)

    print("PER MODEL  (retention over 1,000 words, truncation at 0.80)")
    print(f"  {'projection':10s} {'model':14s} {'peak med':>9s} {'95% CI':>17s} "
          f"{'p10-p90':>15s} {'words up':>9s} {'mean med':>9s}")
    for _, r in M.iterrows():
        print(f"  {r.projection:10s} {r.model:14s} {r.peak_median:9.3f} "
              f"[{r.peak_ci_lo:6.3f},{r.peak_ci_hi:6.3f}] "
              f"{r.peak_p10:7.3f}-{r.peak_p90:6.3f} {100*r.peak_share_up:8.1f}% "
              f"{r.mean_median:9.3f}")

    # ----------------------------------------------------- model level (5c)
    print("\nMODEL-LEVEL TESTS  (one value per model; exploratory)")
    print(f"  {'projection':10s} {'measure':14s} {'mean':>8s} {'pos':>6s} "
          f"{'t p':>9s} {'Wilcoxon p':>11s}")
    out = []
    for proj in sorted(M.projection.unique()):
        sub = M[M.projection == proj].set_index("model").reindex(DEC).dropna(how="all")
        for label, col in (("peak retention", "peak_mean"),
                           ("mean retention", "mean_mean")):
            v = (sub[col] - 1).dropna().values
            if len(v) < 2:
                continue
            t_p = stats.ttest_1samp(v, 0).pvalue
            w_p = stats.wilcoxon(v).pvalue
            print(f"  {proj:10s} {label:14s} {v.mean():+8.4f} "
                  f"{int((v > 0).sum())}/{len(v):<4d} {t_p:9.4f} {w_p:11.4f}")
            out.append(dict(projection=proj, measure=label, n_models=len(v),
                            mean_change=v.mean(), n_positive=int((v > 0).sum()),
                            t_p=t_p, wilcoxon_p=w_p))
    pd.DataFrame(out).to_csv(f"{OUT}/intervention_semcor_model_level.csv", index=False)

    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
