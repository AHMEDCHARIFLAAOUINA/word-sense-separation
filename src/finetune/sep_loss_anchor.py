"""Anchor-based separation loss (first version of the regulariser)."""
import os
import json
import random
from pathlib import Path

import torch

from src.extraction.target_token_extractor import find_target_token_index

MANIFEST = Path("data/raw/semcor_words/_manifest.json")
WORD_DIR = Path("data/raw/semcor_words")
SEED = 0


class AnchorSeparationBatch:
    """Batches of one word: one anchor sentence and B-1 of the other sense."""

    def __init__(self, tokenizer, device, manifest=MANIFEST, max_words=None, exclude=(),
                 batch_size=16, max_length=64, seed=SEED, normalise=False):
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size   # B in the formula
        self.max_length = max_length
        self.normalise = normalise   # L2-normalise W before the distance
        self.rng = random.Random(seed)

        words = [w["word"] for w in json.load(open(manifest))["words"]]
        words = [w for w in words if w not in set(exclude)]
        if max_words:
            words = words[:max_words]

        self.by_word, skipped = {}, 0
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
                by_sense.setdefault(lab, []).append(
                    {"sentence": s, "token": t, "word": w, "label": lab})
            usable = {k: v for k, v in by_sense.items() if len(v) >= 1}
            if len(usable) >= 2 and sum(len(v) for v in usable.values()) >= 3:
                self.by_word[w] = usable
            else:
                skipped += 1

        self.words = sorted(self.by_word)
        self.order = list(self.words)
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.items = [it for senses in self.by_word.values()
                      for v in senses.values() for it in v]
        n_sent = len(self.items)
        print(f"[anchor pool] {len(self.words)} words, {n_sent} sentences, "
              f"{skipped} skipped; B = {batch_size}, one word per step"
              f"{', normalised' if normalise else ''}")

    # ------------------------------------------------------------------ batch
    def next_word(self):
        """Words are taken in turn, reshuffled at the end of each pass."""
        if self.cursor >= len(self.order):
            self.rng.shuffle(self.order)
            self.cursor = 0
        w = self.order[self.cursor]
        self.cursor += 1
        return w

    def sample(self, n=None, rng=None):
        """One batch: the anchor first, then B-1 sentences of the other sense."""
        rng = rng or self.rng
        B = n or self.batch_size
        for _ in range(8):   # a few attempts if a word is thin
            w = self.next_word()
            senses = list(self.by_word[w])
            rng.shuffle(senses)
            anchor_sense, other_sense = senses[0], senses[1]
            anchors = self.by_word[w][anchor_sense]
            others = self.by_word[w][other_sense]
            if not anchors or not others:
                continue
            anchor = rng.choice(anchors)
            k = min(B - 1, len(others))
            rest = rng.sample(others, k)
            if len(rest) >= 1:
                return [anchor] + rest
        return None

    # ------------------------------------------------------------------- loss
    def loss(self, model, batch, layers):
        """L_sem, already carrying its minus sign: add it to the task loss."""
        if not batch or len(batch) < 2:
            return None
        enc = self.tokenizer([b["sentence"] for b in batch], padding=True,
                             truncation=True, max_length=self.max_length,
                             return_tensors="pt").to(self.device)
        out = model(**enc, output_hidden_states=True, return_dict=True)

        terms = []
        for l in layers:
            h = out.hidden_states[l]
            idx = torch.tensor([min(b["token"], h.shape[1] - 1) for b in batch],
                               device=self.device)
            W = h[torch.arange(h.shape[0], device=self.device), idx, :].float()
            if self.normalise:
                W = torch.nn.functional.normalize(W, dim=-1)
            d2 = ((W[0:1] - W[1:]) ** 2).sum(dim=-1)   # ||W_1,1 - W_i,j||^2
            terms.append(d2.sum() / (len(batch) - 1))   # divide by B - 1
        if not terms:
            return None
        return -0.5 * torch.stack(terms).mean()   # the leading -(1/2)

    def separation(self, model, batch, layers):
        """(1/2)*mean squared distance, positive."""
        v = self.loss(model, batch, layers)
        return None if v is None else -v

    def mean_distance(self, model, batch, layers):
        """The same quantity without the sign, for reporting."""
        v = self.loss(model, batch, layers)
        return None if v is None else float(-2.0 * v)


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer",
                    default=os.path.join(os.environ.get("WSS_MODELS", "models"), "Llama/Llama-3-8b"))
    ap.add_argument("--max-words", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    pool = AnchorSeparationBatch(tok, "cpu", max_words=a.max_words,
                                 batch_size=a.batch_size)
    b = pool.sample()
    print(f"\nbatch of {len(b)} sentences, word '{b[0]['word']}'")
    print(f"  anchor  {b[0]['label']:24s} {b[0]['sentence'][:60]}")
    for x in b[1:4]:
        print(f"  other   {x['label']:24s} {x['sentence'][:60]}")
    print(f"  ... {len(b)-1} sentences of sense '{b[1]['label']}'")
