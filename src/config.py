"""Load the per-word dataset configs."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

from src.utils.paths import get_project_root


def config_path(word: str) -> Path:
    return get_project_root() / "configs" / "words" / f"{word}.yaml"


def _parse_simple_yaml(text: str) -> dict:
    """Minimal YAML reader for flat key/value plus simple lists."""
    out, current_list_key = {}, None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line.lstrip().startswith("- ") and current_list_key:
            out[current_list_key].append(line.lstrip()[2:].strip())
            continue
        if ":" in line and not line.startswith(" "):
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            if val in ("", ">", "|"):
                out[key] = [] if val == "" else ""
                current_list_key = key if val == "" else None
                continue
            val = val.strip('"').strip("'")
            if val.isdigit():
                val = int(val)
            out[key] = val
            current_list_key = None
    return out


def load_word_config(word: str) -> dict:
    p = config_path(word)
    if not p.exists():
        raise FileNotFoundError(
            f"No config for word '{word}'. Expected: {p}\n"
            f"Available: {[c.stem for c in p.parent.glob('*.yaml')]}"
        )
    cfg = _parse_simple_yaml(p.read_text(encoding="utf-8"))
    cfg["word"] = cfg.get("word", word)
    cfg["_config_path"] = str(p)
    return cfg


def resolve_dataset(cfg: dict) -> Path:
    ds = get_project_root() / cfg["dataset"]
    if not ds.exists():
        raise FileNotFoundError(f"Dataset not found: {ds}")
    return ds


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_dataset(word: str) -> tuple:
    """Return (data_dict, dataset_path, config)."""
    cfg = load_word_config(word)
    ds = resolve_dataset(cfg)
    with open(ds, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("target_word", "").lower() != cfg["word"].lower():
        raise ValueError(
            f"Config word '{cfg['word']}' does not match dataset "
            f"target_word '{data.get('target_word')}' in {ds}"
        )
    if len(data["sentences"]) != len(data["labels"]):
        raise ValueError("sentences and labels have different lengths")
    return data, ds, cfg
