"""Context-only baseline over many words, with the model loaded once.

    python scripts/geometry_separation/context_baseline_batch.py --models llama-3-8b"""
import argparse, json, os, sys, pathlib, time, traceback
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import src.analysis.context_baseline as cb

_CACHE = {}
if hasattr(cb, "load_model_and_tokenizer"):
    _orig = cb.load_model_and_tokenizer

    def _cached(name, *a, **k):
        if name not in _CACHE:
            _CACHE[name] = _orig(name, *a, **k)
        return _CACHE[name]

    cb.load_model_and_tokenizer = _cached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama-3-8b")
    ap.add_argument("--manifest", default="data/raw/semcor_words/_manifest.json")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer-frac", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="results/analysis/_context_baseline/context_baseline_semcor.csv")
    a = ap.parse_args()

    words = [w["word"] for w in json.load(open(a.manifest))["words"]]
    rng = np.random.default_rng(a.seed)
    if a.limit and a.limit < len(words):
        words = list(rng.choice(words, size=a.limit, replace=False))
    models = [m.strip() for m in a.models.split(",") if m.strip()]

    rows, failures, t0 = [], [], time.time()
    for i, w in enumerate(words, 1):
        for m in models:
            try:
                df = cb.run_model(m, w, a.layer_frac, a.batch_size)
                df["word"] = w
                rows.append(df)
            except Exception as e:
                failures.append((w, m, f"{type(e).__name__}: {e}"))
                if len(failures) <= 3:
                    traceback.print_exc()
        if i % 10 == 0 or i == len(words):
            el = time.time() - t0
            print(f"[{i}/{len(words)}] {el/60:.1f} min, "
                  f"{el/i:.1f} s/word, {len(failures)} failed", flush=True)

    if not rows:
        raise SystemExit("no results; see the traceback above")
    df = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)

    piv = df.pivot_table(index=["model", "word"], columns="representation",
                         values="accuracy")
    print("\nCONTEXT-ONLY BASELINE on the SemCor words")
    print(f"  {'model':22s} {'words':>6s} {'context':>9s} {'target':>8s} "
          f"{'combined':>9s} {'gain':>7s} {'ctx>=0.9':>9s}")
    for m, g in piv.groupby(level=0):
        ctx = g.get("context")
        tgt = g.get("target")
        com = g.get("combined")
        share = (ctx >= 0.9).mean() if ctx is not None else np.nan
        print(f"  {m:22s} {len(g):6d} {ctx.median():9.3f} "
              f"{(tgt.median() if tgt is not None else np.nan):8.3f} "
              f"{(com.median() if com is not None else np.nan):9.3f} "
              f"{((com - ctx).median() if com is not None else np.nan):+7.3f} "
              f"{100*share:8.1f}%")
    print("\n  (medians over words; ctx>=0.9 is the share of words whose senses are "
          "almost fully recoverable without the target)")
    if failures:
        print(f"\n{len(failures)} failures, e.g. {failures[:3]}")
    print("\nwritten:", a.out)


if __name__ == "__main__":
    main()
