"""Tasks, data loading and evaluation for fine-tuning with a separation regulariser
(LoRA). The training used in the paper is sense_pull_regulariser.py."""

from __future__ import annotations

import os
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_dataset as load_word_dataset
from src.models.hf_loader import load_model_and_tokenizer
from src.extraction.target_token_extractor import find_target_token_index
from src.utils.paths import get_results_root, ensure_dir

SEED = 0   # overridden by --seed
DATA = Path(os.environ.get("WSS_BENCH_DATA", "benchmark_data"))
WORDS = ["bank", "bat", "crane", "seal", "plant", "pupil", "club"]


# ── downstream tasks ──────────────────────────────────────────────────────────

def load_wic(rng):
    """Word in context."""
    rows = [json.loads(l) for l in
            open(DATA / "SuperGLUE/WiC/val.jsonl", encoding="utf-8")]
    out = []
    for r in rows:
        prompt = (f"Sentence 1: {r['sentence1']}\n"
                  f"Sentence 2: {r['sentence2']}\n"
                  f"Question: Is the word \"{r['word']}\" used with the same "
                  f"meaning in both sentences?\nAnswer:")
        out.append({"prompt": prompt,
                    "answer": " yes" if r["label"] == "true" else " no",
                    "options": [" yes", " no"],
                    "label": 0 if r["label"] == "true" else 1})
    rng.shuffle(out)
    return out[:450], out[450:]


def load_gsm8k(rng):
    """Grade-school arithmetic with worked solutions."""
    rows = [json.loads(l) for l in
            open(DATA / "gsm8k/train.jsonl", encoding="utf-8")]
    rng.shuffle(rows)
    out = [{"prompt": f"Question: {r['question']}\nAnswer:",
            "answer": " " + r["answer"].split("####")[0].strip(),
            "final": r["answer"].split("####")[-1].strip()}
           for r in rows]
    return out[:1500], out[1500:1800]


def load_winogrande(rng):
    """Pronoun resolution requiring commonsense."""
    rows = [json.loads(l) for l in
            open(DATA / "winogrande/train_l.jsonl", encoding="utf-8")]
    rng.shuffle(rows)
    out = []
    for r in rows:
        s = r["sentence"]
        out.append({"prompt": f"Sentence: {s}\nWhat does _ refer to?\nAnswer:",
                    "answer": " " + r["option" + r["answer"]],
                    "options": [" " + r["option1"], " " + r["option2"]],
                    "label": int(r["answer"]) - 1})
    return out[:1500], out[1500:1800]


def load_triviaqa(rng):
    """Open-domain factual recall."""
    rows = []
    with open(DATA / "triviaqa/triviaqa-train.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            a = r.get("answer")
            a = a[0] if isinstance(a, list) and a else a
            if not isinstance(a, str) or not a.strip():
                continue
            rows.append({"prompt": f"Question: {r['question']}\nAnswer:",
                         "answer": " " + a.strip(), "final": a.strip()})
            if len(rows) >= 2000:
                break
    rng.shuffle(rows)
    return rows[:1500], rows[1500:1800]


def load_gsm8k_full(rng):
    """All 7,473 GSM8K training questions; the official test set (1,319)."""
    def rows(name):
        out = []
        for l in open(DATA / f"gsm8k/{name}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            out.append({"prompt": f"Question: {r['question']}\nAnswer:",
                        "answer": " " + r["answer"].split("####")[0].strip(),
                        "final": r["answer"].split("####")[-1].strip()})
        return out
    train = rows("train")
    rng.shuffle(train)
    return train, rows("test")


TASKS = {"wic": load_wic, "gsm8k": load_gsm8k, "gsm8k_full": load_gsm8k_full,
         "winogrande": load_winogrande, "triviaqa": load_triviaqa}


# ── separation term ───────────────────────────────────────────────────────────

from src.finetune.separation_pool_semcor import SeparationBatchSemCor
from src.finetune.separation_pool_final import SeparationBatchSemCorFinal
from src.finetune.sep_loss_anchor import AnchorSeparationBatch
from src.finetune.sense_probe import FixedSenseProbe


class SeparationBatch:
    """Sentences and target-token indices for the regularisation term."""

    def __init__(self, tokenizer, words, per_word, device, max_length=64):
        self.device = device
        self.items = []
        rng = random.Random(SEED)
        for w in words:
            data, _, _ = load_word_dataset(w)
            idx = list(range(len(data["sentences"])))
            rng.shuffle(idx)
            for i in idx[:per_word]:
                s = data["sentences"][i]
                t, ok = find_target_token_index(tokenizer, s, data["target_word"])
                if t is None or not ok:
                    continue
                self.items.append({"sentence": s, "token": t,
                                   "label": f"{w}:{data['labels'][i]}"})
        self.tokenizer = tokenizer
        self.max_length = max_length

    def sample(self, n, rng):
        return rng.sample(self.items, min(n, len(self.items)))

    def separation(self, model, batch, layers):
        """Differentiable Sep(l), averaged over the given layers."""
        enc = self.tokenizer([b["sentence"] for b in batch], padding=True,
                             truncation=True, max_length=self.max_length,
                             return_tensors="pt").to(self.device)
        out = model(**enc, output_hidden_states=True, return_dict=True)

        labels = [b["label"] for b in batch]
        same = torch.tensor([[a == c for c in labels] for a in labels],
                            device=self.device)
        eye = torch.eye(len(batch), dtype=torch.bool, device=self.device)
        intra, inter = same & ~eye, ~same & ~eye
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
            seps.append(S[intra].mean() - S[inter].mean())
        return torch.stack(seps).mean()


# ── training ──────────────────────────────────────────────────────────────────

class PromptDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=320):
        self.rows, self.tok, self.max_length = rows, tokenizer, max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p = self.tok(r["prompt"], add_special_tokens=True)["input_ids"]
        full = self.tok(r["prompt"] + r["answer"],
                        add_special_tokens=True)["input_ids"][:self.max_length]
        labels = list(full)
        for j in range(min(len(p), len(labels))):
            labels[j] = -100
        return {"input_ids": full, "labels": labels}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids, lab, att = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * k)
        lab.append(b["labels"] + [-100] * k)
        att.append([1] * len(b["input_ids"]) + [0] * k)
    return (torch.tensor(ids), torch.tensor(lab), torch.tensor(att))


@torch.no_grad()
def evaluate(model, tokenizer, rows, device, task):
    """Held-out loss and, for multiple-choice tasks, accuracy."""
    model.eval()
    losses, correct, n = [], 0, 0

    for r in rows:
        p_ids = tokenizer(r["prompt"], return_tensors="pt").input_ids
        f_ids = tokenizer(r["prompt"] + r["answer"], return_tensors="pt").input_ids.to(device)
        if f_ids.shape[1] <= p_ids.shape[1]:
            continue
        lg = model(f_ids).logits[:, :-1, :].float()
        lp = torch.log_softmax(lg, -1).gather(2, f_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        losses.append(-float(lp[0, p_ids.shape[1] - 1:].mean()))

        if "options" in r:
            scores = []
            for o in r["options"]:
                g = tokenizer(r["prompt"] + o, return_tensors="pt").input_ids.to(device)
                l2 = model(g).logits[:, :-1, :].float()
                s2 = torch.log_softmax(l2, -1).gather(2, g[:, 1:].unsqueeze(-1)).squeeze(-1)
                scores.append(float(s2[0, p_ids.shape[1] - 1:].mean()))
            correct += int(int(np.argmax(scores)) == r["label"])
            n += 1

    model.train()
    return {"loss": float(np.mean(losses)) if losses else float("nan"),
            "accuracy": correct / n if n else float("nan")}


def run(a):
    from peft import LoraConfig, get_peft_model
    global SEED
    SEED = a.seed

    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    tokenizer, model = load_model_and_tokenizer(a.model)[:2]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device

    train_rows, test_rows = TASKS[a.task](rng)
    if a.train_n and a.train_n < len(train_rows):
        train_rows = random.Random(1000 + a.seed).sample(train_rows, a.train_n)
    print(f"  {a.task}: {len(train_rows)} train, {len(test_rows)} test")

    probe = FixedSenseProbe(tokenizer, device)
    held_out = set(probe.words) if a.held_out_probe else set()
    if a.sep_pool == "anchor":
        sep_data = AnchorSeparationBatch(tokenizer, device,
                                         batch_size=a.sep_batch,
                                         normalise=a.sep_normalise,
                                         exclude=held_out)
    elif a.sep_pool in ("semcor", "semcor_final"):
        sep_data = (SeparationBatchSemCorFinal if a.sep_pool == "semcor_final" else SeparationBatchSemCor)(
            tokenizer, device, words_per_step=a.sep_words_per_step,
            per_sense=a.sep_per_sense, exclude=held_out)
    else:
        sep_data = SeparationBatch(tokenizer, WORDS, a.sep_per_word, device)
    print(f"  regulariser pool: {len(sep_data.items)} sentences")
    if a.sep_cover:
        from src.finetune.sep_cover import attach_cover
        a.max_steps = attach_cover(sep_data, rng, a.sep_batch, a.seed)
        a.epochs = max(a.epochs, 10000)

    n_layers = model.config.num_hidden_layers
    layers = list(range(n_layers - a.sep_layers + 1, n_layers + 1))
    print(f"  separation measured at layers {layers[0]}-{layers[-1]} of {n_layers}")

    if a.grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        print("  gradient checkpointing on")
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    # fixed evaluation subset: every point on the curve uses the same items
    curve_rng = random.Random(SEED + 7)
    curve_rows = (test_rows if a.eval_n <= 0 or a.eval_n >= len(test_rows)
                  else curve_rng.sample(test_rows, a.eval_n))
    print(f"  periodic evaluation on {len(curve_rows)} items every "
          f"{a.eval_every} steps")

    before = evaluate(model, tokenizer, test_rows, device, a.task)

    # fixed batches for measuring Sep; training still samples fresh ones
    eval_rng = random.Random(SEED + 1)
    sep_eval = [sep_data.sample(a.sep_batch, eval_rng) for _ in range(8)]

    def measure(m):
        with torch.no_grad():
            vals = [sep_data.separation(m, b, layers) for b in sep_eval]
        vals = [float(v) for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")


    def geometry(m):
        return probe.measure(m, layers)

    sep0 = measure(model)
    geo0 = geometry(model)
    print(f"  before training: norm {geo0['norm']:.2f}, same {geo0['same']:.4f}, cross {geo0['cross']:.4f}, cos_sep {geo0['cos_sep']:.4f}")
    print(f"  before training: loss {before['loss']:.4f}, "
          f"accuracy {before['accuracy']:.4f}, Sep {sep0:.4f}")

    loader = DataLoader(PromptDataset(train_rows, tokenizer),
                        generator=torch.Generator().manual_seed(a.seed),
                        batch_size=a.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr)
    model.train()

    step, t0, log = 0, time.time(), []
    point0 = evaluate(model, tokenizer, curve_rows, device, a.task)
    curve = [{"step": 0, "loss": point0["loss"],
              "accuracy": point0["accuracy"], "sep": sep0}]
    for epoch in range(a.epochs):
        for ids, lab, att in loader:
            ids, lab, att = ids.to(device), lab.to(device), att.to(device)
            loss_task = model(input_ids=ids, attention_mask=att, labels=lab).loss

            loss_sep = torch.tensor(0.0, device=device)
            sep_val = float("nan")
            if a.condition != "baseline":
                with torch.random.fork_rng(devices=[device]):
                    s = sep_data.separation(model, sep_data.sample(a.sep_batch, rng), layers)
                if s is not None:
                    sep_val = float(s)
                    if a.condition == "increase":
                        loss_sep = -s
                    elif a.condition == "decrease":
                        loss_sep = s
                    else:
                        loss_sep = (s - sep0).abs()

            loss = loss_task + a.lambda_sep * loss_sep
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()

            step += 1

            if a.eval_every and step % a.eval_every == 0:
                pt = evaluate(model, tokenizer, curve_rows, device, a.task)
                sp = measure(model)
                curve.append({"step": step, "loss": pt["loss"],
                              "accuracy": pt["accuracy"], "sep": sp})
                print(f"    [eval] step {step:4d}  loss {pt['loss']:.4f}"
                      f"  acc {pt['accuracy']:.4f}  Sep {sp:+.4f}")

            if step % a.log_every == 0:
                log.append({"step": step, "loss_task": float(loss_task),
                            "sep": sep_val})
                print(f"    step {step:4d}  task {float(loss_task):.4f}"
                      f"  Sep {sep_val:+.4f}  ({time.time()-t0:.0f}s)"
                      if sep_val == sep_val else
                      f"    step {step:4d}  task {float(loss_task):.4f}"
                      f"  ({time.time()-t0:.0f}s)")
            if a.max_steps and step >= a.max_steps:
                break
        if a.max_steps and step >= a.max_steps:
            break

    after = evaluate(model, tokenizer, test_rows, device, a.task)
    sep1 = measure(model)
    geo1 = geometry(model)
    print(f"  after training: norm {geo1['norm']:.2f}, same {geo1['same']:.4f}, cross {geo1['cross']:.4f}, cos_sep {geo1['cos_sep']:.4f}")

    print(f"\n  after training: loss {after['loss']:.4f}, "
          f"accuracy {after['accuracy']:.4f}, Sep {sep1:.4f}")
    print(f"  loss change {after['loss'] - before['loss']:+.4f}   "
          f"accuracy change {after['accuracy'] - before['accuracy']:+.4f}   "
          f"Sep change {sep1 - sep0:+.4f}")

    out = ensure_dir(get_results_root() / "analysis" / "_finetune")
    tag = (f"{a.model}_{a.task}_{a.condition}"
           f"_lam{a.lambda_sep}_s{a.seed}_st{a.max_steps}"
           + ("" if a.sep_pool == "legacy" else f"_{a.sep_pool}")
           + ("_norm" if a.sep_normalise else "")
           + ("_ho" if a.held_out_probe else "") + ("_cov" if a.sep_cover else "")
           + (f"_n{a.train_n}" if a.train_n else ""))
    json.dump({"geo_before": geo0, "geo_after": geo1, "model": a.model, "task": a.task, "condition": a.condition,
               "lambda_sep": a.lambda_sep, "steps": step,
               "sep_layers": layers, "lr": a.lr,
               "seed": a.seed,
               "loss_before": before["loss"], "loss_after": after["loss"],
               "loss_change": after["loss"] - before["loss"],
               "acc_before": before["accuracy"], "acc_after": after["accuracy"],
               "acc_change": after["accuracy"] - before["accuracy"],
               "sep_before": sep0, "sep_after": sep1,
               "sep_change": sep1 - sep0, "log": log,
               "eval_every": a.eval_every, "eval_n": len(curve_rows),
               "curve": curve},
              open(out / f"{tag}.json", "w"), indent=2)
    print(f"  Saved -> {out / f'{tag}.json'}")
    if a.save_adapter:
        model.save_pretrained(str(Path(a.save_adapter) / tag))
        print(f"  Adapter -> {Path(a.save_adapter) / tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3-8b")
    ap.add_argument("--task", default="wic", choices=list(TASKS))
    ap.add_argument("--condition", default="baseline",
                    choices=["baseline", "increase", "preserve", "decrease"])
    ap.add_argument("--lambda-sep", type=float, default=0.1)
    ap.add_argument("--sep-pool", default="legacy",
                    choices=["legacy", "semcor", "anchor", "semcor_final"],
                    help="legacy: seven words, labels word:sense; "
                         "semcor: 1,000 words, within-word pairs only")
    ap.add_argument("--sep-words-per-step", type=int, default=4)
    ap.add_argument("--train-n", type=int, default=0,
                    help="few-shot: keep only this many training rows (0 = all)")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing: same results, less memory")
    ap.add_argument("--held-out-probe", action="store_true",
                    help="exclude the probe words from the regulariser pool")
    ap.add_argument("--sep-normalise", action="store_true",
                    help="anchor pool: L2-normalise W before the distance")
    ap.add_argument("--sep-per-sense", type=int, default=4)
    ap.add_argument("--sep-layers", type=int, default=4,
                    help="How many final layers the separation term covers")
    ap.add_argument("--sep-batch", type=int, default=16)
    ap.add_argument("--sep-per-word", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--sep-cover", action="store_true",
                    help="per word, batches until all its sentences are used; steps = #batches")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-adapter", default="",
                    help="if set, save the LoRA adapter to DIR/<tag>")
    ap.add_argument("--eval-every", type=int, default=20,
                    help="Evaluate every N steps; 0 disables the curve")
    ap.add_argument("--eval-n", type=int, default=100,
                    help="Items used at each evaluation point; 0 for all")
    a = ap.parse_args()

    print(f"\nFine-tuning with separation regulariser")
    print(f"  model {a.model}   task {a.task}   condition {a.condition}"
          f"   lambda {a.lambda_sep}\n")
    run(a)


if __name__ == "__main__":
    main()
