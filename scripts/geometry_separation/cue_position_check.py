"""Separation when most of the context comes before vs after the target word.

    python scripts/geometry_separation/cue_position_check.py"""
import argparse, json, glob, os, sys, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

DEC = ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b", "qwen-7b",
       "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"]
ENC = ["bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base"]


def sep_from_states(H, labels):
    """Sep(l) for one word: same-sense minus different-sense mean cosine."""
    X = H / (np.linalg.norm(H, axis=-1, keepdims=True) + 1e-12)   # (n, L, d)
    n, L, _ = X.shape
    lab = np.asarray(labels)
    same = lab[:, None] == lab[None, :]
    iu = np.triu_indices(n, k=1)
    out = np.empty(L)
    for l in range(L):
        C = X[:, l, :] @ X[:, l, :].T
        c, s = C[iu], same[iu]
        out[l] = (c[s].mean() if s.any() else np.nan) - \
                 (c[~s].mean() if (~s).any() else np.nan)
    return out


def load(word, model):
    hits = glob.glob(f"results/words/{word}/*/{model}/embeddings/hidden_states.npz")
    if not hits:
        return None
    z = np.load(hits[0], allow_pickle=True)
    key = "hidden_states" if "hidden_states" in z else z.files[0]
    H = z[key]
    lab = json.load(open(f"data/raw/semcor_words/{word}.json"))["labels"]
    return H, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--manifest", default="data/raw/semcor_words/_manifest.json")
    ap.add_argument("--out", default="results/analysis/_cross_word/cue_position_semcor.csv")
    a = ap.parse_args()

    models = a.models.split(",") if a.models else DEC + ENC
    words = [w["word"] for w in json.load(open(a.manifest))["words"]]

    # proxy for cue position: share of the sentence's words before the target
    pool = {w["word"]: w for w in json.load(open(a.manifest))["words"]}
    rows = []
    for model in models:
        peaks_before, peaks_after, n_used = [], [], 0
        for word in words:
            got = load(word, model)
            if got is None:
                continue
            H, lab = got
            d = json.load(open(f"data/raw/semcor_words/{word}.json"))
            sents = d["sentences"]
            tw = d["target_word"].lower()
            frac = []
            for s in sents:
                toks = s.split()
                try:
                    i = next(k for k, t in enumerate(toks)
                             if t.lower().strip(".,;:!?\"'()").startswith(tw))
                except StopIteration:
                    i = len(toks) // 2
                frac.append(i / max(1, len(toks)))
            frac = np.asarray(frac)
            if len(frac) != len(lab) or H.shape[0] != len(lab):
                continue
            q1, q3 = np.percentile(frac, [25, 75])
            late, early = frac >= q3, frac <= q1   # extremes, not the median split
            for mask, store in ((late, peaks_before), (early, peaks_after)):
                sub = np.asarray(lab)[mask]
                if len(set(sub)) < 2 or mask.sum() < 4:
                    continue
                store.append(np.nanmax(sep_from_states(H[mask], sub)))
            n_used += 1
        if peaks_before and peaks_after:
            b, f = np.mean(peaks_before), np.mean(peaks_after)
            rows.append(dict(model=model, arch="decoder" if model in DEC else "encoder",
                             n_words=n_used, sep_more_context_before=b,
                             sep_less_context_before=f, difference=b - f,
                             ratio=b / f if f else np.nan))
            print(f"  {model:22s} {'dec' if model in DEC else 'enc'}  words {n_used:4d}  "
                  f"more-context-before {b:.4f}  less {f:.4f}  diff {b-f:+.4f}  "
                  f"ratio {b/f if f else float('nan'):.2f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    if not df.empty:
        for arch, g in df.groupby("arch"):
            print(f"\n{arch}s: mean difference {g.difference.mean():+.4f}, "
                  f"mean ratio {g.ratio.mean():.2f}, {int((g.difference > 0).sum())}/{len(g)} positive")
    print("\nwritten:", a.out)


if __name__ == "__main__":
    main()
