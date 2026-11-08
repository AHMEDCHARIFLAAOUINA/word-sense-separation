"""Separation term on the final token instead of the target token."""
import torch
import torch.nn.functional as F

from src.finetune.separation_pool_semcor import SeparationBatchSemCor

PROMPT = 'Sentence: {s}\nThe word "{w}" in this sentence means'


class SeparationBatchSemCorFinal(SeparationBatchSemCor):

    def separation(self, model, batch, layers):
        texts = [PROMPT.format(s=b["sentence"], w=b["word"]) for b in batch]
        side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=self.max_length + 24,
                             return_tensors="pt").to(self.device)
        self.tokenizer.padding_side = side
        out = model(**enc, output_hidden_states=True, return_dict=True)
        last = enc["attention_mask"].sum(1) - 1

        words = [b["word"] for b in batch]
        labels = [b["label"] for b in batch]
        n = len(batch)
        same_word = torch.tensor([[words[i] == words[j] for j in range(n)]
                                  for i in range(n)], device=self.device)
        same_sense = torch.tensor([[labels[i] == labels[j] for j in range(n)]
                                   for i in range(n)], device=self.device)
        eye = torch.eye(n, dtype=torch.bool, device=self.device)
        intra = same_word & same_sense & ~eye
        inter = same_word & ~same_sense
        if intra.sum() == 0 or inter.sum() == 0:
            return None

        seps = []
        for l in layers:
            h = out.hidden_states[l]
            v = h[torch.arange(n, device=self.device), last, :]
            v = F.normalize(v.float(), dim=-1)
            S = v @ v.T
            per_word = []
            for w in dict.fromkeys(words):
                m = torch.tensor([x == w for x in words], device=self.device)
                a = intra & m[:, None] & m[None, :]
                b = inter & m[:, None] & m[None, :]
                if a.sum() and b.sum():
                    per_word.append(S[a].mean() - S[b].mean())
            if per_word:
                seps.append(torch.stack(per_word).mean())
        return torch.stack(seps).mean() if seps else None
