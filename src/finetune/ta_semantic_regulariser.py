"""Fine-tuning with the semantic separation regulariser.

L = L_sft + lambda * L_sem, one SemCor batch (one word, anchor + other sense) per step.
    --loss ta       L_sem = -1/2 sum ||W1 - Wi||^2 / (B-1)   (pushes the senses apart)
    --loss ta_pull  the same with the sign reversed       (pulls them together)

    python -m src.finetune.ta_semantic_regulariser --model qwen2-7b --task gsm8k_full \
        --loss ta_pull --span mean --layers 25-28 --seed 0 --tag ta_pull --save-adapter <dir>"""
import os
import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.hf_loader import load_model_and_tokenizer
from src.finetune.sep_regularised_finetune import TASKS, PromptDataset, collate, evaluate
from src.finetune.sense_probe import FixedSenseProbe

WORD_DIR = Path(os.path.join(os.environ.get("WSS_PIPELINE", "."), "data/raw/semcor_words"))
PREFIX = "Example sentence: "   # removed from the semantic sentences with --no-prefix
MAX_LEN = 128   # sentence length limit (tokens); long enough for every sentence


# STEP 1: find the start and end token ids of the semantic word in a sentence
def find_word_span(tokenizer, sentence, word):
    """(start, end) token ids of the word, end excluded; None if it cannot be used."""
    # 1a. where is the word in the text (character positions)?
    matches = list(re.finditer(r"\b" + re.escape(word) + r"\b", sentence, flags=re.IGNORECASE))
    if len(matches) != 1:
        return None   # word missing, or appears more than once: skip
    char_start, char_end = matches[0].start(), matches[0].end()

    # 1b. tokenise; offset_mapping gives the character span of every token
    enc = tokenizer(sentence, truncation=True, max_length=MAX_LEN,
                    return_offsets_mapping=True)

    # 1c. every token that overlaps the word's characters belongs to the word
    ids = [i for i, (s, e) in enumerate(enc["offset_mapping"])
           if e > s and s < char_end and e > char_start]
    if not ids:
        return None   # word cut off by MAX_LEN: skip
    if enc["offset_mapping"][ids[-1]][1] < char_end:
        return None   # word only partly kept: skip
    return ids[0], ids[-1] + 1   # start, end


def load_semantic_data(tokenizer, exclude=(), strip_prefix=False):
    """{word: {sense: [{sentence, start, end}, ...]}} for every usable word."""
    manifest = json.load(open(WORD_DIR / "_manifest.json"))
    data = {}
    for entry in manifest["words"]:
        word = entry["word"]
        if word in exclude:
            continue
        d = json.load(open(WORD_DIR / f"{word}.json"))
        by_sense = {}
        for sentence, sense in zip(d["sentences"], d["labels"]):
            sentence = sentence[len(PREFIX):] if strip_prefix and sentence.startswith(PREFIX) else sentence
            span = find_word_span(tokenizer, sentence, d["target_word"])
            if span is not None:
                by_sense.setdefault(sense, []).append(
                    {"sentence": sentence, "start": span[0], "end": span[1]})
        by_sense = {s: v for s, v in by_sense.items() if len(v) >= 2}
        if len(by_sense) >= 2:
            data[word] = by_sense
    return data


# STEP 2: run the sentences through the model, take H[start:end] on every layer
def get_word_vectors(model, tokenizer, batch, layers, device):
    sentences = [x["sentence"] for x in batch]
    tokenizer.padding_side = "right"   # padding at the end, so token ids stay valid
    enc = tokenizer(sentences, padding=True, truncation=True, max_length=MAX_LEN,
                    return_tensors="pt").to(device)
    out = model(**enc, output_hidden_states=True)
    hs = torch.stack([out.hidden_states[l] for l in layers])   # [n_layers, B, seq_len, hidden]
    return [hs[:, i, x["start"]:x["end"]] for i, x in enumerate(batch)]


# STEP 3: the formula  L_sem = -1/2 * sum_{i=2..B} d(H_1, H_i) / (B-1)
def distance(H1, Hi, mode):
    """Squared distance between two words, on every layer -> shape [n_layers]."""
    if mode == "first":
        return ((H1[:, 0] - Hi[:, 0]) ** 2).sum(-1)
    if mode == "pairwise" and H1.shape[1] == Hi.shape[1]:
        return ((H1 - Hi) ** 2).sum(-1).mean(-1)
    m1 = F.normalize(H1.mean(1), dim=-1)
    mi = F.normalize(Hi.mean(1), dim=-1)
    return ((m1 - mi) ** 2).sum(-1)


def semantic_loss(H, mode="mean", normalise=True):
    """H: list of B tensors (step 2)."""
    H = [h.float() for h in H]
    if normalise:   # length 1; without it the distance grows without limit
        H = [F.normalize(h, dim=-1) for h in H]
    d = torch.stack([distance(H[0], H[i], mode) for i in range(1, len(H))])   # [B-1, n_layers]
    per_layer = d.sum(0) / (len(H) - 1)   # sum over i, divided by B-1
    return -0.5 * per_layer.mean()   # -1/2, averaged over layers


def pushpull_loss(H, mode="mean"):
    V = torch.stack([h.float()[:, 0] if mode == "first" else h.float().mean(1) for h in H], 1)
    V = F.normalize(V, dim=-1)   # [n_layers, B, hidden]
    C = V @ V.transpose(1, 2)   # cosines [n_layers, B, B]
    cross = C[:, 0, 1:].mean(-1)   # anchor vs other sense
    k = V.shape[1] - 1
    if k < 2:
        return cross.mean()   # only one other sentence: push only
    iu = torch.triu_indices(k, k, 1, device=V.device)
    same = C[:, 1:, 1:][:, iu[0], iu[1]].mean(-1)   # other sense vs other sense
    return (cross - same).mean()


# STEP 4: for every word, batches until all its sentences are used
def make_batches(data, B=16, seed=0):
    """One batch = 1 anchor of one sense + up to B-1 sentences of the other sense."""
    rng = random.Random(seed)
    words = sorted(data)
    rng.shuffle(words)
    batches = []
    for word in words:
        senses = {s: rng.sample(v, len(v)) for s, v in data[word].items()}
        unused = {s: list(v) for s, v in senses.items()}
        while any(unused.values()):   # until no unused sentence is left
            other = max(unused, key=lambda s: len(unused[s]))   # sense with most unused
            anchor_sense = [s for s in senses if s != other][0]
            if unused[anchor_sense]:
                anchor = unused[anchor_sense].pop()
            else:   # that sense is used up: reuse one
                anchor = rng.choice(senses[anchor_sense])
            rest, unused[other] = unused[other][:B - 1], unused[other][B - 1:]
            batches.append([anchor] + rest)
    return batches


# STEP 5: training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3-8b")
    ap.add_argument("--task", default="wic", choices=list(TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="all", help="'all' or e.g. '29-32'")
    ap.add_argument("--span", default="pairwise", choices=["first", "mean", "pairwise"],
                    help="how the word's tokens H[start:end] are compared")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--raw", action="store_true", help="no L2 normalisation (diverges)")
    ap.add_argument("--baseline", action="store_true", help="L = L_sft only")
    ap.add_argument("--demo", action="store_true", help="show steps 1-4, no training")
    ap.add_argument("--loss", default="pushpull", choices=["ta", "pushpull", "ta_pull"],
                    help="ta = separation loss; ta_pull = separation loss with the sign reversed (pulls the two senses together)")
    ap.add_argument("--tag", default="", help="name of this variant in the output files")
    ap.add_argument("--save-adapter", default="", help="folder to save the LoRA adapter")
    ap.add_argument("--no-prefix", action="store_true", help="remove 'Example sentence: ' from the semantic sentences")
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tokenizer, model = load_model_and_tokenizer(a.model)[:2]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device

    n = model.config.num_hidden_layers
    layers = (list(range(1, n + 1)) if a.layers == "all"
              else list(range(int(a.layers.split("-")[0]), int(a.layers.split("-")[1]) + 1)))

    # semantic data; the 50 probe words are kept out for evaluation
    probe = FixedSenseProbe(tokenizer, device)
    data = load_semantic_data(tokenizer, exclude=set(probe.words), strip_prefix=a.no_prefix)
    batches = make_batches(data, B=16, seed=a.seed)
    allx = [x for s in data.values() for v in s.values() for x in v]
    multi = sum(1 for x in allx if x["end"] - x["start"] > 1)
    print(f"semantic data: {len(data)} words, {len(allx)} sentences -> {len(batches)} batches")
    print(f"words split into several tokens: {multi} sentences ({100 * multi / len(allx):.1f}%)")
    print(f"layers used: {layers[0]}-{layers[-1]} ({len(layers)} layers), span mode: {a.span}")

    if a.demo:
        b = next(bb for bb in batches if bb[0]["end"] - bb[0]["start"] > 1) if multi else batches[0]
        x = b[0]
        ids = tokenizer(x["sentence"], truncation=True, max_length=MAX_LEN)["input_ids"]
        print(f"\nbatch: {len(b)} sentences of one word")
        print(f"  anchor   : {x['sentence'][:80]}")
        print(f"  start:end: {x['start']}:{x['end']} -> tokens "
              f"{[tokenizer.decode([t]) for t in ids[x['start']:x['end']]]}")
        with torch.no_grad():
            H = get_word_vectors(model, tokenizer, b, layers, device)
            print(f"  H_1 shape: {tuple(H[0].shape)}  (layers, word tokens, hidden size)")
            for m in ("first", "mean", "pairwise"):
                print(f"  L_sem ({m:8s}): {semantic_loss(H, m, not a.raw).item():.4f}")
            print(f"  push-and-pull (mean): {pushpull_loss(H, 'mean').item():.4f}")
        return

    # task data: L_sft (the task split depends on the seed)
    train_rows, test_rows = TASKS[a.task](random.Random(a.seed))
    loader = DataLoader(PromptDataset(train_rows, tokenizer), batch_size=4, shuffle=True,
                        generator=torch.Generator().manual_seed(a.seed),
                        collate_fn=lambda x: collate(x, tokenizer.pad_token_id))

    from peft import LoraConfig, get_peft_model
    if a.task in ("gsm8k", "gsm8k_full"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)

    before = evaluate(model, tokenizer, test_rows, device, a.task)
    geo0 = probe.measure(model, layers)
    print(f"before: accuracy {before['accuracy']:.4f}  loss {before['loss']:.4f}  cos_sep {geo0['cos_sep']:.4f}")

    model.train()
    step, t0, task_iter = 0, time.time(), iter(loader)
    for sem_batch in batches:   # one semantic batch per step
        try:
            ids, lab, att = next(task_iter)
        except StopIteration:   # task data repeats as needed
            task_iter = iter(loader)
            ids, lab, att = next(task_iter)
        ids, lab, att = ids.to(device), lab.to(device), att.to(device)

        loss_sft = model(input_ids=ids, attention_mask=att, labels=lab).loss
        if a.baseline:
            loss_sem = torch.tensor(0.0, device=device)
        else:
            H = get_word_vectors(model, tokenizer, sem_batch, layers, device)   # steps 1-2
            if a.loss == "ta":   # step 3
                loss_sem = semantic_loss(H, a.span, normalise=not a.raw)
            elif a.loss == "ta_pull":   # opposite sign: pulls the anchor towards the other sense
                loss_sem = -semantic_loss(H, a.span, normalise=not a.raw)
            else:
                loss_sem = pushpull_loss(H, a.span)
        loss = loss_sft + a.lam * loss_sem

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        step += 1
        if step % 100 == 0:
            print(f"step {step:5d}/{len(batches)}  L_sft {loss_sft.item():.4f}  "
                  f"L_sem {loss_sem.item():+.4f}  ({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    after = evaluate(model, tokenizer, test_rows, device, a.task)
    geo1 = probe.measure(model, layers)
    print(f"after:  accuracy {after['accuracy']:.4f}  loss {after['loss']:.4f}  cos_sep {geo1['cos_sep']:.4f}")

    name = (f"final_{a.task}_{a.tag}_s{a.seed}.json" if a.tag else
            f"final_{a.task}_{'baseline' if a.baseline else a.loss}_s{a.seed}"
            f"_L{layers[0]}-{layers[-1]}{'' if a.baseline else '_' + a.span}_lam{a.lam}"
            f"{'_raw' if a.raw else ''}.json")
    out = Path("results/analysis/_finetune") / name
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"task": a.task, "seed": a.seed, "layers": layers, "span": a.span, "steps": step,
               "loss_type": "none" if a.baseline else a.loss, "tag": a.tag,
               "baseline": a.baseline, "raw": a.raw, "lambda": a.lam,
               "before": before, "after": after, "geo_before": geo0, "geo_after": geo1},
              open(out, "w"), indent=2)
    if a.save_adapter:
        model.save_pretrained(a.save_adapter)
        print(f"adapter -> {a.save_adapter}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
