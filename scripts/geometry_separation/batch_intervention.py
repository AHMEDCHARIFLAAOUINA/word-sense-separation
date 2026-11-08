"""Spectral intervention over many words, with the model loaded and patched once.

    python scripts/geometry_separation/batch_intervention.py --models llama-3-8b --projections q_proj,v_proj"""
import argparse, gc, json, os, sys, pathlib, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import src.analysis.intervention as iv
from src.models.hf_loader import load_model_and_tokenizer
from src.utils.paths import ensure_dir, get_results_root

DEC = ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b", "qwen-7b",
       "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"]


def snapshot(model, projection):
    layers, _ = iv.get_layers(model)
    return [iv.extract_matrices(l)[projection].detach().clone().cpu()
            for l in layers]


def restore(model, projection, saved):
    layers, _ = iv.get_layers(model)
    with torch.no_grad():
        for l, W0 in zip(layers, saved):
            W = iv.extract_matrices(l)[projection]
            W.copy_(W0.to(W.device, W.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama-3-8b")
    ap.add_argument("--model", default="")   # alias
    ap.add_argument("--projections", default="q_proj")
    ap.add_argument("--ratios", default="1.0,0.80")
    ap.add_argument("--condition", default="truncate")
    ap.add_argument("--manifest", default="data/raw/semcor_words/_manifest.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    models = [a.model] if a.model else [m.strip() for m in a.models.split(",") if m.strip()]
    projections = [p.strip() for p in a.projections.split(",") if p.strip()]
    ratios = [float(r) for r in a.ratios.split(",")]
    words = [w["word"] for w in json.load(open(a.manifest))["words"]]
    if a.limit:
        words = words[:a.limit]

    out_dir = ensure_dir(get_results_root() / "analysis" / "_intervention_semcor")
    out = a.out or str(out_dir / f"intervention_semcor_{a.condition}.csv")

    rows, t0 = [], time.time()
    for model_name in models:
        print(f"\n=== {model_name} ===", flush=True)
        tokenizer, model = load_model_and_tokenizer(model_name)[:2]
        model.eval()
        for projection in projections:
            saved = snapshot(model, projection)
            for ratio in ratios:
                restore(model, projection, saved)
                if ratio < 1.0:
                    st = iv.patch_model(model, projection, ratio, a.condition)
                    sr_b = float(st.before_stable_rank.mean())
                    sr_a = float(st.after_stable_rank.mean())
                    fro = float(1 - st.after_fro_norm.mean() / st.before_fro_norm.mean())
                else:
                    layers, _ = iv.get_layers(model)
                    base = [iv.spectral_stats(iv.extract_matrices(l)[projection])
                            for l in layers]
                    sr_b = sr_a = float(np.mean([b["stable_rank"] for b in base]))
                    fro = 0.0
                print(f"  {projection} ratio {ratio:.2f}: stable rank "
                      f"{sr_b:.1f} -> {sr_a:.1f} ({sr_a/sr_b:.1%})", flush=True)

                t1 = time.time()
                for i, w in enumerate(words, 1):
                    try:
                        sep = iv.separation_after(model, tokenizer, w, a.batch_size)
                    except Exception as e:
                        print(f"    {w}: {type(e).__name__}: {e}", flush=True)
                        continue
                    rows.append(dict(
                        model=model_name, word=w, projection=projection,
                        condition=a.condition, ratio=ratio,
                        stable_rank_before=sr_b, stable_rank_after=sr_a,
                        stable_rank_retained=sr_a / sr_b if sr_b else np.nan,
                        fro_change=fro,
                        sep_max=float(sep.separation.max()),
                        sep_peak_layer=int(sep.separation.idxmax()),
                        sep_mean=float(sep.separation.mean())))
                    if i % 100 == 0 or i == len(words):
                        el = time.time() - t1
                        print(f"    [{i}/{len(words)}] {el/60:.1f} min, "
                              f"{el/i:.2f} s/word", flush=True)
                        pd.DataFrame(rows).to_csv(out, index=False)
            restore(model, projection, saved)
            del saved
            gc.collect()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    # retention against the ratio-1.0 reference of the same model x word x projection
    base = df[df.ratio == 1.0].set_index(["model", "word", "projection"])
    k = ["model", "word", "projection"]
    df["sep_max_retained"] = df.apply(
        lambda r: r.sep_max / base.sep_max.get(tuple(r[c] for c in k), np.nan), axis=1)
    df["sep_mean_retained"] = df.apply(
        lambda r: r.sep_mean / base.sep_mean.get(tuple(r[c] for c in k), np.nan), axis=1)
    df.to_csv(out, index=False)

    print(f"\ntotal {(time.time()-t0)/60:.1f} min, {len(df)} rows -> {out}")
    for (m, p, r), g in df[df.ratio < 1.0].groupby(["model", "projection", "ratio"]):
        print(f"  {m:14s} {p:8s} ratio {r:.2f}  "
              f"peak retention median {g.sep_max_retained.median():.3f}  "
              f"words up {int((g.sep_max_retained > 1).sum())}/{len(g)}")


if __name__ == "__main__":
    main()
