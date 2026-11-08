"""Extraction + metrics for many words with the model loaded once.

    python scripts/geometry_separation/batch_run_words.py --model llama-3-8b"""
import argparse, json, os, time, traceback

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import src.pipeline.extract_embeddings_target as ext
from src.utils.paths import get_standard_result_files
from run import run_one

_CACHE = {}


def _cached_loader(orig):
    def loader(model_name, *a, **k):
        if model_name not in _CACHE:
            _CACHE[model_name] = orig(model_name, *a, **k)
        return _CACHE[model_name]
    return loader


ext.load_model_and_tokenizer = _cached_loader(ext.load_model_and_tokenizer)
try:   # the final-token path, if it is used
    import src.pipeline.extract_embeddings_final_token as extf
    extf.load_model_and_tokenizer = _cached_loader(extf.load_model_and_tokenizer)
except Exception:
    pass


def words_from_manifest(path):
    return [w["word"] for w in json.load(open(path))["words"]]


def done(model, word):
    try:
        f = get_standard_result_files(model, word)
        for key in ("separation_csv", "cosine_metrics_csv"):
            if key in f:
                return os.path.exists(f[key])
        return False
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", default="data/raw/semcor_words/_manifest.json")
    ap.add_argument("--stages", default="extract,metrics")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--redo", action="store_true", help="ignore existing results")
    a = ap.parse_args()

    words = words_from_manifest(a.manifest)[a.start:]
    if a.limit:
        words = words[:a.limit]
    stages = a.stages.split(",")

    t0, failures, skipped = time.time(), [], 0
    for i, w in enumerate(words, 1):
        if not a.redo and done(a.model, w):
            skipped += 1
            continue
        try:
            for st in stages:
                run_one(a.model, w, st, a.batch_size, True, False)
        except Exception as e:
            failures.append((w, f"{type(e).__name__}: {e}"))
            print(f"[{i}/{len(words)}] {w}  FAILED  {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue
        if i % 10 == 0 or i == len(words):
            el = time.time() - t0
            rate = el / max(1, i - skipped)
            print(f"[{i}/{len(words)}] {w}  {el/60:.1f} min elapsed, "
                  f"{rate:.1f} s/word, {rate*(len(words)-i)/60:.0f} min left, "
                  f"{len(failures)} failed", flush=True)

    print(f"\ndone: {len(words)-len(failures)-skipped} run, {skipped} already present, "
          f"{len(failures)} failed, {(time.time()-t0)/60:.1f} min")
    if failures:
        with open(f"logs/semcor_failures_{a.model}.json", "w") as f:
            json.dump(failures, f, indent=1)
        print("failures written to logs/semcor_failures_%s.json" % a.model)
        for w, e in failures[:10]:
            print(" ", w, e)


if __name__ == "__main__":
    main()
