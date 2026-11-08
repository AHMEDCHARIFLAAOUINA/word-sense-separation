"""Hidden states of the target word (first subword token) at every layer."""

import re

import numpy as np
import torch

MIN_TOKEN_INDEX = 3
NORM_RATIO_LIMIT = 10.0


class ExtractionValidationError(RuntimeError):
    """Raised when extracted representations fail a validation gate."""


def _nows_len(s: str) -> int:
    return len(re.sub(r"\s+", "", s))


def find_target_token_index(tokenizer, sentence, target_word, max_length=128):
    """Return (token_index_of_first_subword, validated_flag)."""
    pattern = r"\b" + re.escape(target_word) + r"\b"
    m = re.search(pattern, sentence, flags=re.IGNORECASE)
    if m is None:
        m = re.search(re.escape(target_word), sentence, flags=re.IGNORECASE)
    if m is None:
        return None, False

    start_nows = _nows_len(sentence[: m.start()])

    ids = tokenizer(
        sentence,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    idx = None
    for k in range(1, len(ids) + 1):
        decoded = tokenizer.decode(ids[:k], skip_special_tokens=True)
        if _nows_len(decoded) > start_nows:
            idx = k - 1
            break

    if idx is None:
        return None, False

    piece = tokenizer.decode([ids[idx]], skip_special_tokens=True)
    piece_clean = re.sub(r"\s+", "", piece).lower()
    tw = target_word.lower()
    validated = bool(piece_clean) and (
        tw.startswith(piece_clean) or piece_clean.startswith(tw)
    )
    return idx, validated


def extract_target_token_hidden_states(
    tokenizer,
    model,
    sentences,
    target_word,
    max_length=128,
    batch_size=8,
    strict=True,
    min_token_index=MIN_TOKEN_INDEX,
    norm_ratio_limit=NORM_RATIO_LIMIT,
):
    try:
        tokenizer.padding_side = "right"
    except Exception:
        pass

    indices, validated = [], []
    for s in sentences:
        i, v = find_target_token_index(tokenizer, s, target_word, max_length)
        indices.append(i)
        validated.append(v)

    keep_mask = [i is not None for i in indices]

    # -- gate 0: location ------------------------------------------------------
    missing = [i for i, idx in enumerate(indices) if idx is None]
    if len(missing) == len(sentences):
        raise ExtractionValidationError(
            f"The target word '{target_word}' was not found in any of the "
            f"{len(sentences)} sentences. Check that the dataset's "
            f"target_word matches its sentences."
        )
    if missing and strict:
        examples = "\n".join(f"    [{i}] {sentences[i][:70]}" for i in missing[:5])
        raise ExtractionValidationError(
            f"The target word '{target_word}' was not found in "
            f"{len(missing)}/{len(sentences)} sentences. Dropping them would "
            f"unbalance the classes, so extraction stops. Fix the sentences "
            f"(check spelling and inflected forms).\n{examples}"
        )

    # -- gate 1: position ------------------------------------------------------
    low = [(i, idx) for i, idx in enumerate(indices)
           if idx is not None and idx < min_token_index]
    if low and strict:
        examples = "\n".join(f"    idx={idx} | {sentences[i][:70]}" for i, idx in low[:5])
        raise ExtractionValidationError(
            f"{len(low)}/{len(sentences)} sentences place the target token at "
            f"position < {min_token_index}, where attention-sink activations "
            f"dominate the hidden state. Add a neutral prefix or rewrite these "
            f"sentences.\n{examples}"
        )

    device = next(model.parameters()).device
    collected = []

    for b in range(0, len(sentences), batch_size):
        chunk = sentences[b: b + batch_size]
        chunk_idx = indices[b: b + batch_size]

        enc = tokenizer(chunk, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, return_dict=True)

        hs = torch.stack(out.hidden_states, dim=0).permute(1, 0, 2, 3)
        seq_len = hs.shape[2]

        for j, ti in enumerate(chunk_idx):
            if ti is None:
                continue
            collected.append(hs[j, :, min(ti, seq_len - 1), :].float().cpu().numpy())

        del out, hs
        torch.cuda.empty_cache()

    embeddings = np.stack(collected, axis=0)

    # -- gate 2: norms ---------------------------------------------------------
    mid = embeddings.shape[1] // 2
    norms = np.linalg.norm(embeddings[:, mid, :], axis=1)
    med = float(np.median(norms))
    ratio = float(norms.max() / med) if med > 0 else float("inf")
    if ratio > norm_ratio_limit and strict:
        raise ExtractionValidationError(
            f"Anomalous representation norms at layer {mid}: max/median = "
            f"{ratio:.1f} (limit {norm_ratio_limit}). This usually indicates "
            f"target tokens in attention-sink positions or a tokenisation "
            f"mismatch."
        )

    report = {
        "n_sentences": len(sentences),
        "n_located": int(sum(keep_mask)),
        "n_validated": int(sum(validated)),
        "validation_rate": float(sum(validated)) / max(1, len(sentences)),
        "min_token_index": int(min(i for i in indices if i is not None)),
        "max_token_index": int(max(i for i in indices if i is not None)),
        "n_below_position_gate": len(low),
        "norm_max_over_median": round(ratio, 2),
        "strict": strict,
    }
    return embeddings, keep_mask, report
