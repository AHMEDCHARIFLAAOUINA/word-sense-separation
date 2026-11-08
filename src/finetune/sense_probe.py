"""Sense probe on 50 held-out SemCor words (cos_sep), the same for every run."""
import os
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.extraction.target_token_extractor import find_target_token_index

MANIFEST = Path(os.path.join(os.environ.get("WSS_PIPELINE", "."), "data/raw/semcor_words/_manifest.json"))
WORD_DIR = Path(os.path.join(os.environ.get("WSS_PIPELINE", "."), "data/raw/semcor_words"))


class FixedSenseProbe:
    def __init__(self, tokenizer, device, n_words=50, per_sense=8,
                 seed=12345, max_length=64):
        self.tokenizer, self.device, self.max_length = tokenizer, device, max_length
        rng = random.Random(seed)
        words = [w["word"] for w in json.load(open(MANIFEST))["words"]]
        rng.shuffle(words)
        self.batches, self.words = [], []
        for w in words:
            if len(self.batches) >= n_words:
                break
            try:
                d = json.load(open(WORD_DIR / f"{w}.json"))
            except FileNotFoundError:
                continue
            by_sense = {}
            for s, lab in zip(d["sentences"], d["labels"]):
                t, ok = find_target_token_index(tokenizer, s, d["target_word"])
                if t is not None and ok:
                    by_sense.setdefault(lab, []).append({"sentence": s, "token": t,
                                                         "label": lab})
            senses = [v for v in by_sense.values() if len(v) >= 2]
            if len(senses) < 2:
                continue
            batch = []
            for v in senses[:2]:
                batch += rng.sample(v, min(per_sense, len(v)))
            self.batches.append(batch)
            self.words.append(w)
        print(f"[sense probe] {len(self.batches)} fixed words, "
              f"{sum(len(b) for b in self.batches)} sentences")

    @torch.no_grad()
    def measure(self, model, layers):
        norms, same, cross = [], [], []
        for b in self.batches:
            enc = self.tokenizer([x["sentence"] for x in b], padding=True,
                                 truncation=True, max_length=self.max_length,
                                 return_tensors="pt").to(self.device)
            out = model(**enc, output_hidden_states=True, return_dict=True)
            lab = [x["label"] for x in b]
            n = len(b)
            ss = torch.tensor([[lab[i] == lab[j] for j in range(n)] for i in range(n)],
                              device=self.device)
            eye = torch.eye(n, dtype=torch.bool, device=self.device)
            for l in layers:
                h = out.hidden_states[l]
                idx = torch.tensor([min(x["token"], h.shape[1] - 1) for x in b],
                                   device=self.device)
                W = h[torch.arange(n, device=self.device), idx, :].float()
                norms.append(float(W.norm(dim=-1).mean()))
                C = F.normalize(W, dim=-1) @ F.normalize(W, dim=-1).T
                same.append(float(C[ss & ~eye].mean()))
                cross.append(float(C[~ss].mean()))
        s, c = float(np.mean(same)), float(np.mean(cross))
        return {"norm": float(np.mean(norms)), "same": s, "cross": c, "cos_sep": s - c}
