"""Geometry vs separation for every model and word: partial r with linear or quadratic
depth control, per-model means and model-level tests.

    python -m src.analysis.cross_word --measure stable_rank --projection q_proj \
        --words-file data/raw/semcor_words/_manifest.json --tag semcor"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis import _stats as S   # noqa: E402
from src.analysis._io import ALIGNMENTS, DECODERS, FAMILIES, merge_geometry_sep   # noqa: E402
from src.utils.paths import available_words, ensure_dir, get_results_root   # noqa: E402

COLUMN = {
    "spectral_norm": "{proj}_spectral_norm",
    "fro_norm": "{proj}_fro_norm",
    "erank": "{proj}_erank",
    "stable_rank": "{proj}_stable_rank",
}

# depth control: linear (primary) and quadratic (stress test); a cubic fit would
# absorb most of the variance with 28-36 layers
CONTROLS = ((1, "", "linear"), (2, "_quad", "quadratic"))

# set from the command line; keeps tagged runs in their own files
TAG = ""

GROUPS = {"decoders": DECODERS, "llama": FAMILIES["llama"],
          "qwen": FAMILIES["qwen"], "encoders": FAMILIES["bert"]}


def cell(model, word, measure, proj, alignment):
    """Per-cell statistics and the residual series needed for model-level tests."""
    which = "erank" if measure == "erank" else "params"
    try:
        m = merge_geometry_sep(model, word, which=which, alignment=alignment)
    except ValueError as e:
        print(f"  WARNING  {model}/{word}: {e}  -- skipped")
        return None
    if m is None:
        return None
    col = COLUMN[measure].format(proj=proj)
    if col not in m.columns or m[col].isna().any() or m[col].std() == 0:
        return None

    depth = m["layer"].to_numpy(float)
    g = m[col].to_numpy(float)
    s = m["separation"].to_numpy(float)
    n = len(m)
    out, res = {"n_layers": n}, {}
    for deg, sfx, _ in CONTROLS:
        rg, rs = S.residualise(g, depth, deg), S.residualise(s, depth, deg)
        r, n_eff, p_eff = S.effective_n_p(rg, rs, deg)
        out[f"partial_r{sfx}"] = r
        # naive per-cell p (layers treated as independent), supplementary only
        out[f"partial_p{sfx}_naive"] = float(stats.pearsonr(rg, rs)[1])
        out[f"n_eff{sfx}"] = n_eff
        out[f"partial_p{sfx}_eff"] = p_eff
        out[f"lag1_geometry{sfx}"] = S.lag1(rg)
        out[f"lag1_sep{sfx}"] = S.lag1(rs)
        res[sfx] = (rg, rs)
    out["partial_spearman_r"] = float(stats.spearmanr(*res[""])[0])
    return out, res


def model_table(df, resid):
    rows = []
    for model, by_word in resid.items():
        d = df[df.model == model]
        words = list(by_word)
        row = {"family": d.family.iloc[0], "model": model, "arch": d.arch.iloc[0],
               "n_words": len(words), "n_layers": int(d.n_layers.iloc[0])}
        for _, sfx, _ in CONTROLS:
            rx = by_word[words[0]][sfx][0]   # geometry residuals are shared by all words
            obs, p, _ = S.circular_shift_p(rx, [by_word[w][sfx][1] for w in words])
            r = d[f"partial_r{sfx}"]
            if not np.isclose(obs, r.mean(), atol=1e-9):
                raise RuntimeError(f"{model}: permutation mean {obs} != cell mean {r.mean()}")
            row.update({f"mean_r{sfx}": obs, f"min_r{sfx}": float(r.min()),
                        f"max_r{sfx}": float(r.max()),
                        f"n_words_positive{sfx}": int((r > 0).sum()),
                        f"perm_p{sfx}": p})
        row["mean_spearman_r"] = float(d.partial_spearman_r.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(df, M, words):
    rows = []
    for group, members in GROUPS.items():
        sub = M[M.model.isin(members)]
        if sub.empty:
            continue
        for _, sfx, name in CONTROLS:
            s = S.model_level_summary(dict(zip(sub.model, sub[f"mean_r{sfx}"])))
            rows.append({"group": group, "control": name, "words": "all", **s})
    if len(words) > 1:
        for w in words:
            sub = df[(df.arch == "decoder") & (df.word != w)]
            if sub.empty:
                continue
            for _, sfx, name in CONTROLS:
                vals = sub.groupby("model")[f"partial_r{sfx}"].mean().to_dict()
                s = S.model_level_summary(vals)
                rows.append({"group": "decoders", "control": name, "words": f"without_{w}", **s})
    return pd.DataFrame(rows)


def fmt_p(p):
    return "   n/a" if p is None or not np.isfinite(p) else f"{p:6.4f}"


def run(measure, proj, words, alignment):
    tag = f"{proj} {measure} vs Sep, depth-controlled, alignment={alignment}"
    print(f"\n{'=' * 78}\nCROSS-WORD  --  {tag}\nwords: {', '.join(words)}\n{'=' * 78}")
    print("Estimand: per-model mean of word-level partial r. Cells are descriptive;")
    print("model-level p-values are exploratory (models are not independent samples).")

    rows, resid = [], {}
    for fam, models in FAMILIES.items():
        for model in models:
            for w in words:
                c = cell(model, w, measure, proj, alignment)
                if c is None:
                    continue
                out, res = c
                rows.append({"family": fam, "model": model,
                             "arch": "decoder" if model in DECODERS else "encoder",
                             "word": w, "projection": proj, "measure": measure,
                             "alignment": alignment, **out})
                resid.setdefault(model, {})[w] = res
    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo data.")
        return

    for _, sfx, name in CONTROLS:
        print(f"\n  Partial r per cell, {name} depth control (no significance marks)")
        print(f"  {'model':<22}" + "".join(f"{w:>9}" for w in words))
        print("  " + "-" * (22 + 9 * len(words)))
        for model in [m for ms in FAMILIES.values() for m in ms if m in resid]:
            d = df[df.model == model].set_index("word")
            print(f"  {model:<22}" + "".join(
                f"{d.loc[w, f'partial_r{sfx}']:>+9.3f}" if w in d.index else f"{'--':>9}"
                for w in words))

    M = model_table(df, resid)
    print(f"\n{'=' * 78}\nPER MODEL  (mean r over words [min, max], words positive, circular-shift p)\n{'=' * 78}")
    print(f"  {'model':<20}{'linear':>9}{'range':>17}{'pos':>6}{'perm p':>8}"
          f"{'quad':>9}{'range':>17}{'pos':>6}{'perm p':>8}")
    for _, r in M.iterrows():
        print(f"  {r.model:<20}{r.mean_r:>+9.3f}  [{r.min_r:+.2f},{r.max_r:+.2f}]"
              f"{f'{r.n_words_positive}/{r.n_words}':>6}{r.perm_p:>8.3f}"
              f"{r.mean_r_quad:>+9.3f}  [{r.min_r_quad:+.2f},{r.max_r_quad:+.2f}]"
              f"{f'{r.n_words_positive_quad}/{r.n_words}':>6}{r.perm_p_quad:>8.3f}")
    print("  (circular-shift p cannot fall below 1/n_layers, about 0.03)")

    T = summary_table(df, M, words)
    print(f"\n{'=' * 78}\nMODEL-LEVEL TESTS  (one value per model; exploratory)\n{'=' * 78}")
    print(f"  {'group':<10}{'control':<11}{'mean':>8}{'pos':>7}{'LOO range':>18}{'t p':>9}{'Wilcoxon p':>12}")
    for _, r in T[T.words == "all"].iterrows():
        print(f"  {r.group:<10}{r.control:<11}{r['mean']:>+8.3f}"
              f"{f'{r.n_positive}/{r.n_models}':>7}"
              f"{f'[{r.loo_min:+.3f},{r.loo_max:+.3f}]':>18}"
              f"{fmt_p(r.t_p):>9}{fmt_p(r.wilcoxon_p):>12}")
    print("  (encoders: 4 models, the smallest attainable Wilcoxon p is 0.125)")

    lowo = T[T.words != "all"]
    if not lowo.empty:
        print("\n  Leave-one-word-out, decoders")
        print(f"  {'excluded':<14}{'linear':>8}{'pos':>6}{'Wilcoxon p':>12}{'quad':>9}{'pos':>6}{'Wilcoxon p':>12}")
        for w in words:
            a = lowo[(lowo.words == f"without_{w}") & (lowo.control == "linear")]
            b = lowo[(lowo.words == f"without_{w}") & (lowo.control == "quadratic")]
            if a.empty:
                continue
            a, b = a.iloc[0], b.iloc[0]
            print(f"  {w:<14}{a['mean']:>+8.3f}{f'{a.n_positive}/{a.n_models}':>6}{fmt_p(a.wilcoxon_p):>12}"
                  f"{b['mean']:>+9.3f}{f'{b.n_positive}/{b.n_models}':>6}{fmt_p(b.wilcoxon_p):>12}")

    print("\n  Family means (decoders)")
    print(f"    {'family':<8}{'linear':>9}{'quadratic':>11}{'retained':>10}")
    for fam in ["llama", "qwen"]:
        sub = M[M.family == fam]
        if sub.empty:
            continue
        lin, quad = sub.mean_r.mean(), sub.mean_r_quad.mean()
        print(f"    {fam:<8}{lin:>+9.3f}{quad:>+11.3f}{(quad / lin if lin else 0):>10.0%}")

    dec = df[df.arch == "decoder"]
    print("\n  SENSITIVITY: share of decoder cells with p < 0.05 (supplementary only)")
    print(f"    {'control':<11}{'naive':>8}{'effective-n':>13}{'mean n_eff':>12}")
    for _, sfx, name in CONTROLS:
        print(f"    {name:<11}{(dec[f'partial_p{sfx}_naive'] < .05).mean():>8.1%}"
              f"{(dec[f'partial_p{sfx}_eff'] < .05).mean():>13.1%}"
              f"{dec[f'n_eff{sfx}'].mean():>12.1f}")

    out = ensure_dir(get_results_root() / "analysis" / "_cross_word")
    suffix = ("" if alignment == "post" else f"_{alignment}") + TAG
    base = f"cross_word_{proj}_{measure}{suffix}"
    df.to_csv(out / f"{base}.csv", index=False)
    M.to_csv(out / f"{base}_models.csv", index=False)
    T.to_csv(out / f"{base}_summary.csv", index=False)
    print(f"\nSaved -> {out / base}.csv, _models.csv, _summary.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", default="spectral_norm", choices=list(COLUMN))
    ap.add_argument("--projection", default="q_proj")
    ap.add_argument("--words", default="", help="comma-separated")
    ap.add_argument("--words-file", default="",
                    help="JSON manifest or text file, one word per line")
    ap.add_argument("--tag", default="",
                    help="suffix for the output files, e.g. semcor")
    ap.add_argument("--alignment", default="post", choices=list(ALIGNMENTS))
    a = ap.parse_args()

    if a.words_file:
        import json
        if a.words_file.endswith(".json"):
            d = json.load(open(a.words_file))
            words = [w["word"] for w in d["words"]] if isinstance(d, dict) else list(d)
        else:
            words = [l.strip() for l in open(a.words_file) if l.strip()]
    elif a.words:
        words = [w.strip() for w in a.words.split(",") if w.strip()]
    else:
        raise SystemExit(
            "Give --words or --words-file. The default over every folder in "
            "results/words/ would mix the controlled words with the SemCor ones."
        )

    have = set(available_words())
    missing = [w for w in words if w not in have]
    words = [w for w in words if w in have]
    if missing:
        print(f"{len(missing)} words have no results and are skipped, "
              f"e.g. {missing[:5]}")
    if not words:
        raise SystemExit("None of the requested words have results.")
    print(f"{len(words)} words, alignment={a.alignment}, tag={a.tag or '(none)'}")

    TAG = f"_{a.tag}" if a.tag else ""
    run(a.measure, a.projection, words, a.alignment)
