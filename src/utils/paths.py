"""Paths of the data, results and model folders."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

VALID_FAMILIES = {"llama", "bert", "qwen"}

DEFAULT_WORD = "bank"


def get_project_root() -> Path:
    """Project root, assuming this file is at project_root/src/utils/paths.py."""
    return Path(__file__).resolve().parents[2]


def infer_family(model_name: str) -> str:
    """Infer model family from the model identifier."""
    name = model_name.strip().lower()

    if name.startswith("llama"):
        return "llama"

    if name.startswith((
        "bert", "roberta", "deberta", "modernbert",
        "xlm-roberta", "gte", "spanbert",
    )):
        return "bert"

    if name.startswith("qwen"):
        return "qwen"

    raise ValueError(
        f"Could not infer family from model name: '{model_name}'. "
        f"Expected a name starting with one of: llama, bert, roberta, "
        f"deberta, modernbert, xlm-roberta, gte, spanbert, qwen."
    )


def get_results_root() -> Path:
    """Global results root, overridable via SEMANTIC_RESULTS_ROOT."""
    override = os.environ.get("SEMANTIC_RESULTS_ROOT")
    if override:
        return Path(override)
    return get_project_root() / "results"


# -- parameter geometry (word-independent) ------------------------------------

def get_parameters_dir(model_name: str) -> Path:
    """results/parameters/{family}/{model}/"""
    return get_results_root() / "parameters" / infer_family(model_name) / model_name


def get_parameter_files(model_name: str) -> Dict[str, Path]:
    d = get_parameters_dir(model_name)
    return {
        "root": d,
        "effective_rank": d / "effective_rank.csv",
        "parameter_stats": d / "parameter_stats.csv",
    }


# -- per-word results ---------------------------------------------------------

def get_word_root(word: str) -> Path:
    """results/words/{word}/"""
    return get_results_root() / "words" / word


def get_model_result_dir(model_name: str, word: str = DEFAULT_WORD) -> Path:
    """results/words/{word}/{family}/{model}/"""
    return get_word_root(word) / infer_family(model_name) / model_name


def get_model_paths(model_name: str, word: str = DEFAULT_WORD) -> Dict[str, Path]:
    """All canonical output directories for a (model, word) pair."""
    root = get_model_result_dir(model_name, word)
    return {
        "root": root,
        "embeddings": root / "embeddings",
        "metrics": root / "metrics",
        "plots": root / "plots",
        "pca": root / "plots" / "pca",
        "heatmaps": root / "plots" / "heatmaps",
        "trends": root / "plots" / "trends",
        "summaries": root / "summaries",
        "logs": root / "logs",
    }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_model_result_dirs(model_name: str, word: str = DEFAULT_WORD) -> Dict[str, Path]:
    paths = get_model_paths(model_name, word)
    for p in paths.values():
        ensure_dir(p)
    return paths


def get_standard_result_files(model_name: str, word: str = DEFAULT_WORD) -> Dict[str, Path]:
    """Standard output file paths for a (model, word) pair."""
    p = get_model_paths(model_name, word)
    return {
        "embedding_tensor": p["embeddings"] / "hidden_states.npz",
        "embedding_metadata": p["embeddings"] / "embedding_metadata.json",
        "labels": p["embeddings"] / "labels.json",
        "sentences": p["embeddings"] / "sentences.json",
        "run_manifest": p["root"] / "run_manifest.json",
        "cosine_metrics_csv": p["metrics"] / "cosine_layerwise.csv",
        "l2_metrics_csv": p["metrics"] / "l2_layerwise.csv",
        "drift_metrics_csv": p["metrics"] / "layer_drift.csv",
        "separation_csv": p["metrics"] / "semantic_separation.csv",
        "layer_ranking_csv": p["summaries"] / "layer_ranking.csv",
        "top_layers_json": p["summaries"] / "top_layers.json",
        "run_log": p["logs"] / "run.log",
        "run_summary_json": p["summaries"] / "run_summary.json",
    }


# -- cross-word / cross-model analysis outputs --------------------------------

def get_analysis_dir() -> Path:
    return ensure_dir(get_results_root() / "analysis")


def get_analysis_plot_dir(subdir: str = "") -> Path:
    d = get_analysis_dir() / "plots"
    if subdir:
        d = d / subdir
    return ensure_dir(d)


# -- discovery helpers --------------------------------------------------------

def available_words() -> list:
    """Words that have at least one result directory."""
    root = get_results_root() / "words"
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def available_models(word: str = DEFAULT_WORD) -> list:
    """Models with results for a given word."""
    root = get_word_root(word)
    if not root.exists():
        return []
    out = []
    for fam in sorted(root.iterdir()):
        if fam.is_dir():
            out.extend(sorted(m.name for m in fam.iterdir() if m.is_dir()))
    return out
