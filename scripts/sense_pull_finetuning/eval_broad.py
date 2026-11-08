"""Evaluate one model on the other benchmarks. Multiple choice by mean log-probability per token;
TyDiQA and TheoremQA by generation.

    python scripts/sense_pull_finetuning/eval_broad.py --model qwen2-7b --adapter <dir>"""
import os
import argparse, csv, json, random, re, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_accuracy as ga   # noqa: E402  (ROOT, generate, trivia_correct)

D = Path(os.environ.get("WSS_BENCH_DATA", "benchmark_data"))
OUT = ga.ROOT / "results/analysis/_finetune/broad_eval"
N = 1000
MATH_SUBJ = ["abstract_algebra", "college_mathematics", "elementary_mathematics",
             "high_school_mathematics", "high_school_statistics"]
ALL = ["arc_c", "hellaswag", "piqa", "siqa", "csqa", "obqa", "mmlu", "mmlu_math", "strategyqa",
       "tydiqa", "theoremqa"]


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def sample(rows, n=N):
    rows = list(rows)
    random.Random(0).shuffle(rows)
    return rows[:n] if n else rows


def mc(prompt, options, gold):
    return {"prompt": prompt, "options": options, "gold": gold}


def load(b):
    if b == "arc_c":
        out = []
        for r in jl(D / "ARC/ARC-c/ARC-Challenge-Test.jsonl"):
            ch = r["question"]["choices"]; labels = [c["label"] for c in ch]
            if r["answerKey"] in labels:
                out.append(mc(f"Question: {r['question']['stem']}\nAnswer:", [" " + c["text"] for c in ch],
                              labels.index(r["answerKey"])))
        return out
    if b == "obqa":
        out = []
        for r in jl(D / "openbookqa/Main/test.jsonl"):
            ch = r["question"]["choices"]; labels = [c["label"] for c in ch]
            out.append(mc(f"Question: {r['question']['stem']}\nAnswer:", [" " + c["text"] for c in ch],
                          labels.index(r["answerKey"])))
        return out
    if b == "csqa":
        out = []
        for r in jl(D / "commonsenseqa/dev_rand_split.jsonl"):
            ch = r["question"]["choices"]; labels = [c["label"] for c in ch]
            out.append(mc(f"Question: {r['question']['stem']}\nAnswer:", [" " + c["text"] for c in ch],
                          labels.index(r["answerKey"])))
        return sample(out)
    if b == "hellaswag":
        return sample([mc(r["query"], [" " + c for c in r["choices"]], int(r["gold"]))
                       for r in jl(D / "hellaswag/hellaswag.jsonl")])
    if b == "piqa":
        labels = [int(x) for x in open(D / "piqa/dev-labels.lst").read().split()]
        return sample([mc(f"Question: {r['goal']}\nAnswer:", [" " + r["sol1"], " " + r["sol2"]], y)
                       for r, y in zip(jl(D / "piqa/dev.jsonl"), labels)])
    if b == "siqa":
        labels = [int(x) - 1 for x in open(D / "siqa/dev-labels.lst").read().split()]
        return sample([mc(f"{r['context']} Question: {r['question']}\nAnswer:",
                          [" " + r["answerA"], " " + r["answerB"], " " + r["answerC"]], y)
                       for r, y in zip(jl(D / "siqa/dev.jsonl"), labels)])
    if b in ("mmlu", "mmlu_math"):
        files = ([D / f"mmlu/test/{s}_test.csv" for s in MATH_SUBJ] if b == "mmlu_math"
                 else sorted((D / "mmlu/test").glob("*_test.csv")))
        out = []
        for f in files:
            for row in csv.reader(open(f, encoding="utf-8")):
                if len(row) < 6 or row[5].strip() not in ("A", "B", "C", "D"):
                    continue
                q = row[0] + "".join(f"\n{l}. {t}" for l, t in zip("ABCD", row[1:5])) + "\nAnswer:"
                out.append(mc(q, [" A", " B", " C", " D"], "ABCD".index(row[5].strip())))
        return out if b == "mmlu_math" else sample(out)
    if b == "strategyqa":
        rows = json.load(open(D / "strategyqa/strategyQA_train.json"))
        return sample([mc(f"Question: {r['question']}\nAnswer (yes or no):", [" yes", " no"],
                          0 if r["answer"] else 1) for r in rows])
    if b == "tydiqa":
        return [{"prompt": f"{r['passage_text']}\nQuestion: {r['question_text']}\nAnswer:",
                 "gold": [a["text"] for a in r["answers"]]} for r in jl(D / "tydiqa/dev/english-dev.jsonl")]
    if b == "theoremqa":
        out = []
        for r in csv.DictReader(open(D / "TheoremQA/test.csv", encoding="utf-8")):
            pic = (r.get("Picture") or "").strip()
            if r.get("Answer_type") in ("integer", "float") and pic in ("", "nan", "None", "null"):
                try:
                    g = float(r["Answer"])
                except ValueError:
                    continue
                out.append({"prompt": f"Question: {r['Question']}\nAnswer:", "gold": [g]})
        return out
    raise ValueError(b)


@torch.no_grad()
def score_mc(model, tok, items, device, bs=16, max_len=512):
    """Mean log-prob per token of every option; returns predicted index per item."""
    seqs = []   # (item, option, ctx_ids, cont_ids)
    for i, it in enumerate(items):
        ctx = tok(it["prompt"])["input_ids"]
        for k, o in enumerate(it["options"]):
            cont = tok(o, add_special_tokens=False)["input_ids"]
            seqs.append((i, k, ctx, cont))
    seqs.sort(key=lambda x: len(x[2]) + len(x[3]))
    scores = {}
    pad = tok.pad_token_id
    for s in range(0, len(seqs), bs):
        chunk = seqs[s:s + bs]
        full = [(c + o)[-max_len:] for _, _, c, o in chunk]
        L = max(len(f) for f in full)
        ids = torch.full((len(full), L), pad, dtype=torch.long)
        att = torch.zeros((len(full), L), dtype=torch.long)
        for r, f in enumerate(full):
            ids[r, :len(f)] = torch.tensor(f); att[r, :len(f)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
        for r, (i, k, c, o) in enumerate(chunk):
            n = len(full[r]); m = min(len(o), n - 1)
            pos = torch.arange(n - m - 1, n - 1, device=device)   # positions predicting the option
            tgt = ids[r, n - m:n].to(device)
            lp = torch.log_softmax(logits[r, pos].float(), dim=-1)   # only the option positions
            scores[(i, k)] = lp[torch.arange(m, device=device), tgt].mean().item()
        del logits
    return [int(np.argmax([scores[(i, k)] for k in range(len(it["options"]))])) for i, it in enumerate(items)]


def num_close(pred, gold):
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", re.sub(r"<<[^>]*>>", "", pred))
    if not nums:
        return False
    try:
        p = float(nums[-1].replace(",", ""))
    except ValueError:
        return False
    return abs(p - gold) <= max(1e-3, 0.01 * abs(gold))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--benches", default=",".join(ALL))
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N items per benchmark")
    ap.add_argument("--model", default="llama-3-8b", help="base model of the adapter")
    a = ap.parse_args()

    from peft import PeftModel
    from src.models.hf_loader import load_model_and_tokenizer

    tok, model = load_model_and_tokenizer(a.model)[:2]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    name = "pretrained" if a.model == "llama-3-8b" else f"pretrained_{a.model}"
    if a.adapter != "none":
        model = PeftModel.from_pretrained(model, a.adapter)
        name = Path(a.adapter).name
    model.eval()
    device = next(model.parameters()).device
    benches = [b for b in a.benches.split(",") if b]
    if a.adapter != "none" and "_gsm8k_" not in name:
        benches = [b for b in benches if b != "theoremqa"]   # long generation: GSM8K models + pretrained only

    OUT.mkdir(parents=True, exist_ok=True)
    for b in benches:
        out = OUT / f"{b}__{name}{'_limit' if a.limit else ''}.csv"
        if out.exists() and not a.limit:
            print(f"{b}: exists, skipped", flush=True)
            continue
        items = load(b)
        if a.limit:
            items = items[:a.limit]
        if "options" in items[0]:
            tok.padding_side = "right"
            pred = score_mc(model, tok, items, device)
            ok = [p == it["gold"] for p, it in zip(pred, items)]
        else:
            tok.padding_side = "left"
            preds = ga.generate(model, tok, [it["prompt"] for it in items], device,
                                256 if b == "theoremqa" else 24)
            ok = ([num_close(p, it["gold"][0]) for p, it in zip(preds, items)] if b == "theoremqa"
                  else [any(ga.trivia_correct(p, g) for g in it["gold"]) for p, it in zip(preds, items)])
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["item", "correct"]); w.writerows([(i, int(c)) for i, c in enumerate(ok)])
        print(f"{name} {b}: accuracy {np.mean(ok):.4f} ({sum(ok)}/{len(ok)})", flush=True)
    print(f"ALL DONE {name}", flush=True)


if __name__ == "__main__":
    main()
