"""Read geometry and separation results and pair block l with hidden state l+1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.paths import (
    get_parameters_dir, get_model_result_dir, get_analysis_dir, ensure_dir,
)

FAMILIES = {
    "llama": ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b"],
    "qwen": ["qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"],
    "bert": ["bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base"],
}

ALL_MODELS = FAMILIES["llama"] + FAMILIES["qwen"] + FAMILIES["bert"]
DECODERS = FAMILIES["llama"] + FAMILIES["qwen"]
# primary projections: q_proj and k_proj (attention scores Q K^T), v_proj (value path),
# up_proj (feed-forward expansion, present in every architecture)
PRIMARY_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "up_proj"]

# supplementary; gate_proj does not exist in the BERT-family encoders
SUPPLEMENTARY_PROJECTIONS = ["o_proj", "gate_proj", "down_proj"]

ALL_PROJECTIONS = PRIMARY_PROJECTIONS + SUPPLEMENTARY_PROJECTIONS

# Backwards-compatible default
PROJECTIONS = PRIMARY_PROJECTIONS


def sep_path(model: str, word: str) -> Path:
    return get_model_result_dir(model, word) / "metrics" / "semantic_separation.csv"


def load_sep(model: str, word: str):
    p = sep_path(model, word)
    if not p.exists():
        return None
    return pd.read_csv(p).sort_values("layer").reset_index(drop=True)


def load_erank(model: str):
    p = get_parameters_dir(model) / "effective_rank.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).sort_values("layer").reset_index(drop=True)


def load_params(model: str):
    p = get_parameters_dir(model) / "parameter_stats.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).sort_values("layer").reset_index(drop=True)


ALIGNMENTS = ("post", "pre")


def align_geometry_sep(geo, sep, alignment: str = "post"):
    """Pair per-block geometry with per-hidden-state separation."""
    if alignment not in ALIGNMENTS:
        raise ValueError(f"alignment must be one of {ALIGNMENTS}, got {alignment!r}")
    g_idx = geo["layer"].astype(int).tolist()
    s_idx = sep["layer"].astype(int).tolist()
    if g_idx != list(range(len(g_idx))):
        raise ValueError(f"geometry layers not contiguous from 0: {g_idx[:6]}")
    if s_idx != list(range(len(s_idx))):
        raise ValueError(f"separation layers not contiguous from 0: {s_idx[:6]}")
    if len(s_idx) != len(g_idx) + 1:
        raise ValueError(
            f"expected {len(g_idx) + 1} hidden states for {len(g_idx)} blocks "
            f"(embedding + one per block), got {len(s_idx)}")
    shift = 1 if alignment == "post" else 0
    g = geo.copy()
    g["hidden_layer"] = g["layer"].astype(int) + shift
    s = sep.rename(columns={"layer": "hidden_layer"})
    s["hidden_layer"] = s["hidden_layer"].astype(int)
    return g.merge(s, on="hidden_layer", how="inner", validate="one_to_one")


def merge_geometry_sep(model: str, word: str, which: str = "erank",
                       alignment: str = "post"):
    """Merge block geometry with Sep(l)."""
    geo = load_erank(model) if which == "erank" else load_params(model)
    sep = load_sep(model, word)
    if geo is None or sep is None:
        return None
    m = align_geometry_sep(geo, sep, alignment=alignment)
    return m if len(m) >= 5 else None


def analysis_dir(word: str, subdir: str = "") -> Path:
    d = get_analysis_dir() / word
    if subdir:
        d = d / subdir
    return ensure_dir(d)


def analysis_plot_dir(word: str, subdir: str = "") -> Path:
    return analysis_dir(word, "plots" if not subdir else f"plots/{subdir}")


def star(p) -> str:
    if p != p:
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def available(word: str) -> list:
    """Models that have Sep(l) for this word."""
    return [m for m in ALL_MODELS if sep_path(m, word).exists()]
