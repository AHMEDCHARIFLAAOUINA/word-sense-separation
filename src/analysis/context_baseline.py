"""Context-only baseline: classify the sense with the target word removed."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_dataset
from src.models.hf_loader import load_model_and_tokenizer
from src.extraction.target_token_extractor import find_target_token_index
from src.utils.paths import get_model_result_dir, ensure_dir

SEED = 0

ALL_MODELS = [
    "llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b",
    "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b",
    "bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base",
]


def strip_target(sentence, target_word):
    """Remove the target word, leaving the rest of the sentence intact."""
    out = re.sub(rf"\s*\b{re.escape(target_word)}\b", "", sentence, count=1,
                 flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", out).strip()


def mean_pool(model, tokenizer, sentences, layer, batch_size=8, max_length=128):
    """Mean-pooled hidden state at one layer, over non-padding tokens."""
    device = next(model.parameters()).device
    out = []
    for b in range(0, len(sentences), batch_size):
        enc = tokenizer(sentences[b:b + batch_size], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            o = model(**enc, output_hidden_states=True, return_dict=True)
        h = o.hidden_states[layer]
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(pooled.float().cpu().numpy())
        del o, h
        torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def target_states(model, tokenizer, sentences, target_word, layer,
                  batch_size=8, max_length=128):
    """Hidden state at the target token's first subword, at one layer."""
    try:
        tokenizer.padding_side = "right"
    except Exception:
        pass
    idx = [find_target_token_index(tokenizer, s, target_word)[0] for s in sentences]
    device = next(model.parameters()).device
    out, keep = [], []
    for b in range(0, len(sentences), batch_size):
        chunk = sentences[b:b + batch_size]
        cidx = idx[b:b + batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            o = model(**enc, output_hidden_states=True, return_dict=True)
        h = o.hidden_states[layer]
        for j, ti in enumerate(cidx):
            if ti is None:
                keep.append(False)
                continue
            keep.append(True)
            out.append(h[j, min(ti, h.shape[1] - 1), :].float().cpu().numpy())
        del o, h
        torch.cuda.empty_cache()
    return np.stack(out, axis=0), keep


def score(X, y, groups):
    """Cross-validated accuracy of a linear classifier."""
    n_groups = len(set(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2 or len(set(y)) < 2:
        return float("nan"), float("nan")
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=SEED))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    s = cross_val_score(clf, X, y, cv=cv, groups=groups, scoring="accuracy")
    return float(s.mean()), float(s.std())


def run_model(model_name, word, layer_frac=0.5, batch_size=8):
    data, _, _ = load_dataset(word)
    sentences, labels = data["sentences"], data["labels"]
    tw = data["target_word"]
    strata = data.get("stratum")

    tokenizer, model = load_model_and_tokenizer(model_name)[:2]
    model.eval()

    n_layers = model.config.num_hidden_layers
    layer = max(1, min(n_layers, int(round(n_layers * layer_frac))))

    stripped = [strip_target(s, tw) for s in sentences]
    Xc = mean_pool(model, tokenizer, stripped, layer, batch_size)
    Xt, keep = target_states(model, tokenizer, sentences, tw, layer, batch_size)

    groups_all = np.array([i // 2 for i in range(len(sentences))])
    groups = groups_all[np.array(keep)]
    y = np.array([l for l, k in zip(labels, keep) if k])
    Xc = Xc[np.array(keep)]
    Xk = np.concatenate([Xc, Xt], axis=1)

    rows = []
    for name, X in [("context", Xc), ("target", Xt), ("combined", Xk)]:
        m, sd = score(X, y, groups)
        rows.append({"model": model_name, "word": word, "layer": layer,
                     "n_layers": n_layers, "stratum": "all",
                     "representation": name, "n": len(y),
                     "accuracy": m, "accuracy_sd": sd})

    if strata:
        st = np.array([s for s, k in zip(strata, keep) if k])
        for s_name in sorted(set(st)):
            m_ = st == s_name
            if m_.sum() < 40 or len(set(y[m_])) < 2 or len(set(groups[m_])) < 4:
                print(f"    [skip] stratum '{s_name}': "
                      f"{m_.sum()} sentences is too few for grouped folds")
                continue
            for name, X in [("context", Xc), ("target", Xt), ("combined", Xk)]:
                m, sd = score(X[m_], y[m_], groups[m_])
                rows.append({"model": model_name, "word": word, "layer": layer,
                             "n_layers": n_layers, "stratum": s_name,
                             "representation": name, "n": int(m_.sum()),
                             "accuracy": m, "accuracy_sd": sd})

    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def report(df, word):
    print(f"\n{'=' * 74}")
    print(f"CONTEXT-ONLY BASELINE  --  '{word}'")
    print(f"{'=' * 74}")
    for stratum in sorted(df.stratum.unique()):
        d = df[df.stratum == stratum]
        print(f"\n  stratum: {stratum}")
        print(f"    {'model':<22}{'context':>10}{'target':>10}{'combined':>11}{'gain':>9}")
        print("    " + "-" * 62)
        for m in d.model.unique():
            r = d[d.model == m].set_index("representation").accuracy
            gain = r.get("combined", np.nan) - r.get("context", np.nan)
            print(f"    {m:<22}{r.get('context', np.nan):>10.3f}"
                  f"{r.get('target', np.nan):>10.3f}{r.get('combined', np.nan):>11.3f}"
                  f"{gain:>+9.3f}")
        piv = d.pivot_table(index="model", columns="representation", values="accuracy")
        if {"context", "combined"} <= set(piv.columns):
            print(f"    {'mean':<22}{piv['context'].mean():>10.3f}"
                  f"{piv.get('target', pd.Series(dtype=float)).mean():>10.3f}"
                  f"{piv['combined'].mean():>11.3f}"
                  f"{(piv['combined'] - piv['context']).mean():>+9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="bank")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model")
    g.add_argument("--all-models", action="store_true")
    ap.add_argument("--layer-frac", type=float, default=0.5,
                    help="Layer as a fraction of depth; 0.5 is mid-network")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip", default="")
    a = ap.parse_args()

    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    models = [m for m in ALL_MODELS if m not in skip] if a.all_models else [a.model]

    frames = []
    for i, m in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {m}")
        try:
            frames.append(run_model(m, a.word, a.layer_frac, a.batch_size))
        except Exception as e:
            print(f"  FAILED {m}: {type(e).__name__}: {e}")

    if not frames:
        raise SystemExit("no results")
    df = pd.concat(frames, ignore_index=True)
    report(df, a.word)

    out = ensure_dir(get_model_result_dir(models[0], a.word).parents[1]
                     / "_context_baseline")
    path = out / f"context_baseline_{a.word}.csv"
    df.to_csv(path, index=False)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
