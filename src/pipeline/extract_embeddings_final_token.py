"""Extraction stage for datasets without a target word: last-token hidden states."""

import json
import platform
from datetime import datetime, timezone

import numpy as np
import torch

from src.config import load_word_config, resolve_dataset, dataset_sha256
from src.models.hf_loader import load_model_and_tokenizer
from src.utils.paths import ensure_model_result_dirs, get_standard_result_files


def _versions():
    v = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "numpy"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    return v


def extract_final_token(tokenizer, model, sentences, max_length=128, batch_size=8):
    try:
        tokenizer.padding_side = "right"
    except Exception:
        pass

    device = next(model.parameters()).device
    collected = []

    for b in range(0, len(sentences), batch_size):
        chunk = sentences[b:b + batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, return_dict=True)

        hs = torch.stack(out.hidden_states, dim=0).permute(1, 0, 2, 3)
        lengths = enc["attention_mask"].sum(dim=1) - 1

        for j in range(hs.shape[0]):
            collected.append(hs[j, :, int(lengths[j]), :].float().cpu().numpy())

        del out, hs
        torch.cuda.empty_cache()

    emb = np.stack(collected, axis=0)
    mid = emb.shape[1] // 2
    norms = np.linalg.norm(emb[:, mid, :], axis=1)
    ratio = float(norms.max() / np.median(norms)) if np.median(norms) > 0 else float("inf")

    return emb, {"n_sentences": len(sentences),
                 "norm_max_over_median": round(ratio, 2)}


def run_extraction_final_token(model_name: str, word: str, batch_size: int = 8):
    ensure_model_result_dirs(model_name, word)
    files = get_standard_result_files(model_name, word)

    cfg = load_word_config(word)
    ds = resolve_dataset(cfg)
    data = json.load(open(ds, encoding="utf-8"))
    sentences, labels = data["sentences"], data["labels"]

    tokenizer, model = load_model_and_tokenizer(model_name)[:2]
    model.eval()

    emb, report = extract_final_token(tokenizer, model, sentences,
                                      batch_size=batch_size)

    np.savez_compressed(files["embedding_tensor"], embeddings=emb)
    json.dump({"labels": labels}, open(files["labels"], "w"), indent=2)
    json.dump({"sentences": sentences}, open(files["sentences"], "w"),
              indent=2, ensure_ascii=False)
    if "groups" in data:
        json.dump({"groups": data["groups"]},
                  open(files["embeddings"] if False else
                       files["embedding_tensor"].parent / "groups.json", "w"), indent=2)

    json.dump({"model_name": model_name, "word": word,
               "pooling": "final_non_padding_token",
               "num_sentences": int(emb.shape[0]),
               "num_layers": int(emb.shape[1]),
               "hidden_size": int(emb.shape[2]),
               "location_report": report},
              open(files["embedding_metadata"], "w"), indent=2)

    json.dump({"model_name": model_name, "word": word, "dataset": str(ds),
               "dataset_sha256": dataset_sha256(ds),
               "extraction": "final_non_padding_token",
               "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "host": platform.node(), "versions": _versions(),
               "validation": report},
              open(files["run_manifest"], "w"), indent=2)

    print(f"[{model_name}/{word}] {emb.shape[0]} statements, "
          f"norm ratio {report['norm_max_over_median']}")
    print(f"  -> {files['embedding_tensor']}  shape={emb.shape}")

    del model
    torch.cuda.empty_cache()
