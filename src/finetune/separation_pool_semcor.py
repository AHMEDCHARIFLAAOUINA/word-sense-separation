"""Within-word separation term over the SemCor words."""
import os
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from src.extraction.target_token_extractor import find_target_token_index

MANIFEST = Path("data/raw/semcor_words/_manifest.json")
WORD_DIR = Path("data/raw/semcor_words")
SEED = 0


class SeparationBatchSemCor:
    """Within-word Sep(l) over a large word pool."""

    def __init__(self, tokenizer, device, manifest=MANIFEST, max_words=None, exclude=(),
                 words_per_step=4, per_sense=4, max_length=64, seed=SEED):
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.words_per_step = words_per_step
        self.per_sense = per_sense

        rng = random.Random(seed)
        words = [w["word"] for w in json.load(open(manifest))["words"]]
        words = [w for w in words if w not in set(exclude)]
        if max_words:
            words = words[:max_words]

        self.by_word = {}
        skipped = 0
        for w in words:
            try:
                d = json.load(open(WORD_DIR / f"{w}.json"))
            except FileNotFoundError:
                skipped += 1
                continue
            by_sense = {}
            for s, lab in zip(d["sentences"], d["labels"]):
                t, ok = find_target_token_index(tokenizer, s, d["target_word"])
                if t is None or not ok:
                    continue
                by_sense.setdefault(lab, []).append({"sentence": s, "token": t,
                                                     "word": w, "label": lab})
            # a word is usable only with two senses and a pair available in each
            usable = {k: v for k, v in by_sense.items() if len(v) >= 2}
            if len(usable) >= 2:
                self.by_word[w] = usable
            else:
                skipped += 1

        self.words = sorted(self.by_word)
        self.items = [it for senses in self.by_word.values()
                      for v in senses.values() for it in v]
        n_sent = len(self.items)
        print(f"[separation pool] {len(self.words)} words, {n_sent} sentences, "
              f"{skipped} words skipped; {words_per_step} words x "
              f"{per_sense} sentences per sense per step")

    # ------------------------------------------------------------------ sample
    def sample(self, n, rng):
        chosen = rng.sample(self.words, min(self.words_per_step, len(self.words)))
        batch = []
        for w in chosen:
            senses = self.by_word[w]
            for lab in list(senses)[:2]:
                pool = senses[lab]
                k = min(self.per_sense, len(pool))
                batch.extend(rng.sample(pool, k))
        return batch

    # -------------------------------------------------------------- separation
    def separation(self, model, batch, layers):
        """Differentiable within-word Sep(l), averaged over layers and words."""
        enc = self.tokenizer([b["sentence"] for b in batch], padding=True,
                             truncation=True, max_length=self.max_length,
                             return_tensors="pt").to(self.device)
        out = model(**enc, output_hidden_states=True, return_dict=True)

        words = [b["word"] for b in batch]
        labels = [b["label"] for b in batch]
        n = len(batch)
        same_word = torch.tensor([[words[i] == words[j] for j in range(n)]
                                  for i in range(n)], device=self.device)
        same_sense = torch.tensor([[labels[i] == labels[j] for j in range(n)]
                                   for i in range(n)], device=self.device)
        eye = torch.eye(n, dtype=torch.bool, device=self.device)
        intra = same_word & same_sense & ~eye   # same word, same sense
        inter = same_word & ~same_sense   # same word, other sense
        if intra.sum() == 0 or inter.sum() == 0:
            return None

        seps = []
        for l in layers:
            h = out.hidden_states[l]
            idx = torch.tensor([min(b["token"], h.shape[1] - 1) for b in batch],
                               device=self.device)
            v = h[torch.arange(h.shape[0], device=self.device), idx, :]
            v = F.normalize(v.float(), dim=-1)
            S = v @ v.T
            per_word = []
            for w in dict.fromkeys(words):   # order-preserving
                m = torch.tensor([x == w for x in words], device=self.device)
                a = intra & m[:, None] & m[None, :]
                b = inter & m[:, None] & m[None, :]
                if a.sum() and b.sum():
                    per_word.append(S[a].mean() - S[b].mean())
            if per_word:
                seps.append(torch.stack(per_word).mean())
        return torch.stack(seps).mean() if seps else None


if __name__ == "__main__":   # quick check, no GPU needed for the pool itself
    import argparse
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=os.path.join(os.environ.get("WSS_MODELS", "models"), "Llama/Llama-3-8b"))
    ap.add_argument("--max-words", type=int, default=50)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    pool = SeparationBatchSemCor(tok, "cpu", max_words=a.max_words)
    rng = random.Random(0)
    b = pool.sample(0, rng)
    print(f"sampled batch: {len(b)} sentences, "
          f"{len({x['word'] for x in b})} words, "
          f"{len({(x['word'], x['label']) for x in b})} word-sense groups")
    for x in b[:4]:
        print(f"   {x['word']:14s} {x['label']:22s} tok {x['token']:3d}  "
              f"{x['sentence'][:60]}")
