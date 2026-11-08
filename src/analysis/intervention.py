"""Spectral intervention on a projection matrix (truncate / rotate / noise) and its
effect on the separation curve."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_dataset
from src.models.hf_loader import load_model_and_tokenizer
from src.analysis.compute_geometry import get_layers, extract_matrices
from src.extraction.target_token_extractor import extract_target_token_hidden_states
from src.metrics.semantic_separation import compute_semantic_separation
from src.utils.paths import get_results_root, ensure_dir

SEED = 0


def spectral_stats(W: torch.Tensor) -> dict:
    S = torch.linalg.svdvals(W.detach().to(torch.float32)).cpu().numpy()
    Sp = S[S > 1e-10]
    p = Sp / Sp.sum()
    return {
        "spectral_norm": float(S[0]),
        "fro_norm": float(np.sqrt((S ** 2).sum())),
        "stable_rank": float((S ** 2).sum() / (S[0] ** 2)),
        "erank": float(np.exp(-np.sum(p * np.log(p + 1e-12)))),
        "rank_kept": int(len(S)),
    }


def truncate(W: torch.Tensor, ratio: float) -> tuple:
    """Keep the leading directions holding `ratio` of the squared spectrum."""
    W32 = W.detach().to(torch.float32)
    U, S, Vh = torch.linalg.svd(W32, full_matrices=False)
    energy = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
    k = int(torch.searchsorted(energy, torch.tensor(ratio, device=energy.device)).item()) + 1
    k = max(1, min(k, S.numel()))
    Wr = (U[:, :k] * S[:k]) @ Vh[:k, :]
    return Wr.to(W.dtype), k


def rotate(W: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Replace the singular vectors with a random orthonormal basis."""
    W32 = W.detach().to(torch.float32)
    U, S, Vh = torch.linalg.svd(W32, full_matrices=False)
    a = torch.randn(U.shape[0], U.shape[1], generator=generator,
                    device=W32.device, dtype=W32.dtype)
    b = torch.randn(Vh.shape[0], Vh.shape[1], generator=generator,
                    device=W32.device, dtype=W32.dtype)
    Qu, _ = torch.linalg.qr(a)
    Qv, _ = torch.linalg.qr(b.T)
    Wr = (Qu * S) @ Qv.T
    return Wr.to(W.dtype)


def add_noise(W: torch.Tensor, target_delta: float,
              generator: torch.Generator) -> torch.Tensor:
    """Add Gaussian noise whose Frobenius norm matches `target_delta`."""
    W32 = W.detach().to(torch.float32)
    n = torch.randn(W32.shape, generator=generator,
                    device=W32.device, dtype=W32.dtype)
    n = n / torch.linalg.norm(n) * target_delta
    return (W32 + n).to(W.dtype)


def patch_model(model, projection: str, ratio: float, condition: str) -> pd.DataFrame:
    """Modify one projection in every layer, returning before/after statistics."""
    layers, _ = get_layers(model)
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    rows = []

    for i, layer in enumerate(layers):
        mats = extract_matrices(layer)
        if projection not in mats:
            continue
        W = mats[projection]
        before = spectral_stats(W)

        if condition == "truncate":
            new, k = truncate(W, ratio)
        elif condition == "rotate":
            g = torch.Generator(device=W.device.type).manual_seed(SEED + i)
            new, k = rotate(W, g), before["rank_kept"]
        elif condition == "noise":
            ref, _ = truncate(W, ratio)
            delta = float(torch.linalg.norm(
                (W.detach().to(torch.float32) - ref.to(torch.float32))))
            g = torch.Generator(device=W.device.type).manual_seed(SEED + i)
            new, k = add_noise(W, delta, g), before["rank_kept"]
        else:
            raise ValueError(f"unknown condition: {condition}")

        with torch.no_grad():
            W.copy_(new)

        after = spectral_stats(W)
        rows.append({"layer": i, "k_kept": k,
                     **{f"before_{a}": b for a, b in before.items()},
                     **{f"after_{a}": b for a, b in after.items()}})

    return pd.DataFrame(rows)


def separation_after(model, tokenizer, word: str, batch_size: int = 8) -> pd.DataFrame:
    data, _, _ = load_dataset(word)
    emb, keep, _ = extract_target_token_hidden_states(
        tokenizer, model, data["sentences"], data["target_word"],
        batch_size=batch_size, strict=False)
    labels = [l for l, k in zip(data["labels"], keep) if k]
    norm = np.linalg.norm(emb, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    sep = compute_semantic_separation(emb / norm, labels)
    return pd.DataFrame({"layer": np.arange(len(sep)), "separation": sep})


def run(model_name, word, projection, ratios, condition, batch_size):
    out_dir = ensure_dir(get_results_root() / "analysis" / "_intervention")
    records, seps = [], []

    for ratio in ratios:
        print(f"\n  {condition}  {projection}  ratio {ratio:.2f}")
        tokenizer, model = load_model_and_tokenizer(model_name)[:2]
        model.eval()

        # Ratio 1.0 is the unmodified reference for every condition,
        # so that each run contains its own baseline.
        if ratio < 1.0:
            stats = patch_model(model, projection, ratio, condition)
            sr_before = stats.before_stable_rank.mean()
            sr_after = stats.after_stable_rank.mean()
            fro_change = (1 - stats.after_fro_norm.mean() / stats.before_fro_norm.mean())
            print(f"    stable rank {sr_before:.1f} -> {sr_after:.1f}"
                  f"   ({sr_after / sr_before:.1%} retained)"
                  f"   Frobenius change {fro_change:.1%}"
                  f"   mean k {stats.k_kept.mean():.0f}")
        else:
            layers, _ = get_layers(model)
            base = [spectral_stats(extract_matrices(l)[projection]) for l in layers]
            sr_before = sr_after = float(np.mean([b["stable_rank"] for b in base]))
            fro_change = 0.0
            print(f"    unmodified reference, stable rank {sr_before:.1f}")

        sep = separation_after(model, tokenizer, word, batch_size)
        sep["ratio"] = ratio
        seps.append(sep)

        records.append({
            "model": model_name, "word": word, "projection": projection,
            "condition": condition, "ratio": ratio,
            "stable_rank_before": sr_before, "stable_rank_after": sr_after,
            "stable_rank_retained": sr_after / sr_before if sr_before else np.nan,
            "fro_change": fro_change,
            "sep_max": float(sep.separation.max()),
            "sep_peak_layer": int(sep.separation.idxmax()),
            "sep_mean": float(sep.separation.mean()),
        })
        print(f"    Sep max {records[-1]['sep_max']:.4f}"
              f" at layer {records[-1]['sep_peak_layer']}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    tag = f"{model_name}_{word}_{projection}_{condition}"
    df.to_csv(out_dir / f"intervention_{tag}.csv", index=False)
    pd.concat(seps).to_csv(out_dir / f"separation_curves_{tag}.csv", index=False)

    print(f"\n  {'ratio':>7}{'stable rank':>13}{'retained':>10}"
          f"{'Sep max':>10}{'vs baseline':>13}")
    print("  " + "-" * 53)
    base = df.sep_max.iloc[0]
    for _, r in df.iterrows():
        print(f"  {r.ratio:>7.2f}{r.stable_rank_after:>13.1f}"
              f"{r.stable_rank_retained:>10.1%}{r.sep_max:>10.4f}"
              f"{r.sep_max / base if base else np.nan:>13.1%}")
    print(f"\n  Saved -> {out_dir / f'intervention_{tag}.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3-8b")
    ap.add_argument("--word", default="pupil")
    ap.add_argument("--projection", default="q_proj")
    ap.add_argument("--condition", default="truncate",
                    choices=["truncate", "rotate", "noise"])
    ap.add_argument("--ratios", default="1.0,0.95,0.90,0.80")
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()

    ratios = [float(x) for x in a.ratios.split(",")]
    print(f"\nIntervention   {a.model}   word={a.word}   "
          f"{a.projection}   condition={a.condition}")
    run(a.model, a.word, a.projection, ratios, a.condition, a.batch_size)


if __name__ == "__main__":
    main()
