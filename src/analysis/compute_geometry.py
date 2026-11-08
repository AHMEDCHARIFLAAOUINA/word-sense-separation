"""Spectral measures of every projection matrix (exact SVD): spectral norm,
Frobenius norm, effective rank, stable rank.

    python -m src.analysis.compute_geometry --all-models --device cuda:0"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.hf_loader import load_model_and_tokenizer
from src.utils.paths import get_parameters_dir, ensure_dir

PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]

ALL_MODELS = [
    "llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b",
    "qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b",
    "bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base",
]


def get_layers(model):
    """Return (layer_container, pattern_name)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers, "model.layers"
    if hasattr(model, "layers"):
        return model.layers, "layers"
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer, "encoder.layer"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h, "transformer.h"
    if hasattr(model, "base_model"):
        return get_layers(model.base_model)
    raise ValueError(f"Unrecognised layer container in {type(model).__name__}")


def extract_matrices(layer) -> dict:
    """Map projection name -> weight tensor for one layer."""
    m = {}

    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
        a = layer.self_attn
        m["q_proj"] = a.q_proj.weight
        m["k_proj"] = a.k_proj.weight
        m["v_proj"] = a.v_proj.weight
        if hasattr(a, "o_proj"):
            m["o_proj"] = a.o_proj.weight

    elif hasattr(layer, "attn") and hasattr(layer.attn, "c_attn"):
        W = layer.attn.c_attn.weight
        d = W.shape[0] // 3
        m["q_proj"] = W[0:d, :]
        m["k_proj"] = W[d:2 * d, :]
        m["v_proj"] = W[2 * d:, :]
        if hasattr(layer.attn, "c_proj"):
            m["o_proj"] = layer.attn.c_proj.weight

    elif hasattr(layer, "attention") and hasattr(layer.attention, "self"):
        s = layer.attention.self
        for key, names in [("q_proj", ("query_proj", "query")),
                           ("k_proj", ("key_proj", "key")),
                           ("v_proj", ("value_proj", "value"))]:
            for n in names:
                if hasattr(s, n):
                    m[key] = getattr(s, n).weight
                    break
        if hasattr(layer.attention, "output") and hasattr(layer.attention.output, "dense"):
            m["o_proj"] = layer.attention.output.dense.weight

    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        if hasattr(mlp, "gate_proj"):
            m["gate_proj"] = mlp.gate_proj.weight
            m["up_proj"] = mlp.up_proj.weight
            m["down_proj"] = mlp.down_proj.weight
        elif hasattr(mlp, "w1") and hasattr(mlp, "w2"):
            m["up_proj"] = mlp.w1.weight
            m["gate_proj"] = mlp.w2.weight
            if hasattr(mlp, "c_proj"):
                m["down_proj"] = mlp.c_proj.weight

    if hasattr(layer, "intermediate") and hasattr(layer.intermediate, "dense"):
        m["up_proj"] = layer.intermediate.dense.weight
    if hasattr(layer, "output") and hasattr(layer.output, "dense") and "down_proj" not in m:
        m["down_proj"] = layer.output.dense.weight

    return m


def geometry(matrix: torch.Tensor, tol: float = 1e-10) -> dict:
    with torch.no_grad():
        W = matrix.detach().to(torch.float32)
        try:
            S = torch.linalg.svdvals(W).cpu().numpy()
        except Exception:
            S = np.linalg.svd(W.cpu().numpy(), compute_uv=False)

    spectral = float(S[0]) if S.size else float("nan")
    frobenius = float(np.sqrt((S ** 2).sum())) if S.size else float("nan")

    # stable rank = ||A||_F^2 / ||A||_2^2
    stable = float((S ** 2).sum() / (S[0] ** 2)) if S.size and S[0] > 0 else float("nan")

    Sp = S[S > tol]
    if Sp.size == 0:
        erank = 1.0
    else:
        p = Sp / Sp.sum()
        erank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))

    return {"spectral_norm": spectral, "fro_norm": frobenius,
            "erank": erank, "stable_rank": stable}


def _versions() -> dict:
    v = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "numpy"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    return v


def compute_model(model_name: str, cache: bool = False, device=None):
    out_dir = ensure_dir(get_parameters_dir(model_name))
    erank_csv = out_dir / "effective_rank.csv"
    stats_csv = out_dir / "parameter_stats.csv"

    if cache and erank_csv.exists() and stats_csv.exists():
        print(f"  [cached] {model_name}")
        return pd.read_csv(erank_csv)

    print(f"\nLoading {model_name}")
    # one device: sharding over GPUs changes the SVDs at the 1e-6 level
    model = load_model_and_tokenizer(model_name, device=device)[1]
    model.eval()

    layers, pattern = get_layers(model)
    n = len(layers)
    print(f"  container: {pattern}   layers: {n}")

    erank_rows, stats_rows, shapes = [], [], {}
    for i, layer in enumerate(layers):
        mats = extract_matrices(layer)
        er = {"layer": i}
        st = {"layer": i}
        for proj in PROJECTIONS:
            if proj not in mats:
                continue
            g = geometry(mats[proj])
            er[f"{proj}_erank"] = g["erank"]
            st[f"{proj}_spectral_norm"] = g["spectral_norm"]
            st[f"{proj}_fro_norm"] = g["fro_norm"]
            st[f"{proj}_stable_rank"] = g["stable_rank"]
            if i == 0:
                shapes[proj] = list(mats[proj].shape)
        erank_rows.append(er)
        stats_rows.append(st)
        if (i + 1) % 8 == 0 or i == n - 1:
            print(f"    layer {i + 1}/{n}")

    erank_df = pd.DataFrame(erank_rows)
    stats_df = pd.DataFrame(stats_rows)
    erank_df.to_csv(erank_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)

    json.dump({
        "model_name": model_name,
        "layer_container": pattern,
        "n_layers": n,
        "projections": sorted(shapes.keys()),
        "layer0_shapes": shapes,
        "method": "exact SVD (torch.linalg.svdvals), float32",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "versions": _versions(),
    }, open(out_dir / "geometry_manifest.json", "w"), indent=2)

    print(f"  projections: {sorted(shapes.keys())}")
    print(f"  -> {erank_csv}")
    print(f"  -> {stats_csv}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return erank_df


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", type=str)
    g.add_argument("--all-models", action="store_true")
    ap.add_argument("--skip", type=str, default="")
    ap.add_argument("--cache", action="store_true",
                    help="Skip models whose output already exists")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--device", default="cuda:0",
                    help="Single device for weight loading; pass 'auto' to shard")
    a = ap.parse_args()

    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    models = [m for m in ALL_MODELS if m not in skip] if a.all_models else [a.model]

    failed = []
    for i, m in enumerate(models, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(models)}] {m}\n{'=' * 70}")
        try:
            compute_model(m, cache=a.cache,
                          device=None if a.device == 'auto' else a.device)
        except Exception as e:
            failed.append((m, f"{type(e).__name__}: {e}"))
            print(f"  !! FAILED {m}: {type(e).__name__}: {e}")
            if not a.continue_on_error:
                raise

    if failed:
        print("\nFailures:")
        for m, msg in failed:
            print(f"  {m}: {msg[:100]}")
        sys.exit(1)
    print(f"\nDone. {len(models)} model(s).")


if __name__ == "__main__":
    main()
