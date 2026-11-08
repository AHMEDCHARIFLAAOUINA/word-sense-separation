"""Extraction stage: target-token hidden states for every sentence of a word."""

import json
import platform
from datetime import datetime, timezone

import numpy as np

from src.config import load_dataset, dataset_sha256
from src.models.hf_loader import load_model_and_tokenizer
from src.extraction.target_token_extractor import extract_target_token_hidden_states
from src.utils.paths import ensure_model_result_dirs, get_standard_result_files


def _versions():
    v = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "numpy", "scipy", "sklearn"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    return v


def run_extraction_target(model_name: str, word: str = "bank",
                          batch_size: int = 8, strict: bool = True):
    ensure_model_result_dirs(model_name, word)
    files = get_standard_result_files(model_name, word)

    data, ds_path, cfg = load_dataset(word)
    sentences, labels = data["sentences"], data["labels"]
    target_word = data["target_word"]

    expected = cfg.get("senses")
    seen = sorted(set(labels))
    if expected and sorted(expected) != seen:
        raise ValueError(
            f"Labels in the dataset {seen} do not match the senses declared "
            f"in {cfg.get('_config_path')}: {sorted(expected)}"
        )
    if len(seen) != 2:
        raise ValueError(
            f"Sep(l) is defined for exactly two senses; found {len(seen)}: {seen}"
        )
    counts = {s: labels.count(s) for s in seen}
    if min(counts.values()) != max(counts.values()):
        raise ValueError(
            f"Sense classes are unbalanced: {counts}. Sep(l) compares intra- "
            f"and inter-class means, so classes must be equal in size."
        )

    result = load_model_and_tokenizer(model_name)
    tokenizer, model = result[0], result[1]

    embeddings, keep_mask, report = extract_target_token_hidden_states(
        tokenizer, model, sentences, target_word,
        batch_size=batch_size, strict=strict,
    )

    kept_sentences = [s for s, k in zip(sentences, keep_mask) if k]
    kept_labels = [l for l, k in zip(labels, keep_mask) if k]

    np.savez_compressed(files["embedding_tensor"], embeddings=embeddings)
    json.dump({"labels": kept_labels}, open(files["labels"], "w"), indent=2)
    json.dump({"sentences": kept_sentences}, open(files["sentences"], "w"),
              indent=2, ensure_ascii=False)

    json.dump({
        "model_name": model_name,
        "word": word,
        "target_word": target_word,
        "pooling": "target_token_first_subword",
        "num_sentences": int(embeddings.shape[0]),
        "num_layers": int(embeddings.shape[1]),
        "hidden_size": int(embeddings.shape[2]),
        "location_report": report,
    }, open(files["embedding_metadata"], "w"), indent=2)

    json.dump({
        "model_name": model_name,
        "word": word,
        "dataset": str(ds_path),
        "dataset_sha256": dataset_sha256(ds_path),
        "config": cfg.get("_config_path"),
        "extraction": "target_token_first_subword",
        "strict_validation": strict,
        "batch_size": batch_size,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "versions": _versions(),
        "validation": report,
    }, open(files["run_manifest"], "w"), indent=2)

    print(f"[{model_name}/{word}] located {report['n_located']}/{report['n_sentences']}, "
          f"validated {report['n_validated']} ({report['validation_rate']:.1%}), "
          f"token idx {report['min_token_index']}-{report['max_token_index']}, "
          f"norm ratio {report['norm_max_over_median']}")
    print(f"  -> {files['embedding_tensor']}  shape={embeddings.shape}")
