"""Generation helpers and answer checks (GSM8K last number, TriviaQA containment)."""
import argparse, csv, random, re, string, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.finetune.sep_regularised_finetune as ft   # noqa: E402

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def to_num(s):
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def gsm8k_correct(pred, gold):
    pred = re.sub(r"<<[^>]*>>", "", pred)
    nums = NUM.findall(pred)
    g = to_num(gold)
    return bool(nums) and g is not None and to_num(nums[-1]) == g


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def trivia_correct(pred, gold):
    p, g = norm(pred.strip().split("\n")[0]), norm(gold)
    return bool(g) and bool(p) and (g in p or p == g)


def version_of(tag):
    if "_semcor_final" in tag:
        v = "ppfinal"
    elif "_norm" in tag:
        v = "anchor"
    elif "_semcor" in tag:
        v = "pushpull"
    elif "_baseline_" in tag:
        v = "baseline"
    else:
        v = "other"
    return v + ("_cov" if "_cov" in tag else "")


@torch.no_grad()
def generate(model, tok, prompts, device, max_new, bs=16):
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        for row in g[:, enc["input_ids"].shape[1]:]:
            t = tok.decode(row, skip_special_tokens=True)
            outs.append(t.split("\nQuestion:")[0])
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["gsm8k", "triviaqa"])
    ap.add_argument("--match", default="_cov")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from peft import PeftModel
    from src.models.hf_loader import load_model_and_tokenizer

    tok, base = load_model_and_tokenizer("llama-3-8b")[:2]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base.eval()
    device = next(base.parameters()).device

    adapters = sorted(p for p in (ROOT / "adapters").glob(f"*_{a.task}_*")
                      if p.is_dir() and a.match in p.name)
    print(f"{a.task}: {len(adapters)} adapters matching '{a.match}'", flush=True)
    assert adapters, "no adapters found"
    pm = PeftModel.from_pretrained(base, str(adapters[0]), adapter_name="a0")
    for i, p in enumerate(adapters[1:], 1):
        pm.load_adapter(str(p), adapter_name=f"a{i}")
    pm.eval()

    max_new = 256 if a.task == "gsm8k" else 24
    check = gsm8k_correct if a.task == "gsm8k" else trivia_correct
    res, dump = {}, []
    for i, p in enumerate(adapters):
        m = re.search(r"_s(\d+)_st", p.name)
        seed, ver = int(m.group(1)), version_of(p.name)
        _, test = ft.TASKS[a.task](random.Random(seed))
        if a.limit:
            test = test[:a.limit]
        pm.set_adapter(f"a{i}")
        preds = generate(pm, tok, [r["prompt"] for r in test], device, max_new)
        ok = [check(pr, r["final"]) for pr, r in zip(preds, test)]
        res[(ver, seed)] = float(np.mean(ok))
        dump += [{"version": ver, "seed": seed, "gold": r["final"], "pred": pr[:300], "correct": c}
                 for pr, r, c in zip(preds, test, ok)]
        print(f"  {ver:13s} s{seed}  accuracy {res[(ver, seed)]:.4f}  ({sum(ok)}/{len(ok)})", flush=True)

    out = ROOT / f"results/analysis/_finetune/gen_accuracy_{a.task}{a.match}{'_lim' if a.limit else ''}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dump[0])); w.writeheader(); w.writerows(dump)

    print(f"\n=== {a.task} generation accuracy (own-seed test split) ===")
    vers = sorted({v for v, _ in res}, key=lambda v: ("baseline" not in v, v))
    for v in vers:
        vals = [res[(v, s)] for s in (0, 1, 2) if (v, s) in res]
        ref = "baseline_cov" if v.endswith("_cov") else "baseline"
        d = [res[(v, s)] - res[(ref, s)] for s in (0, 1, 2) if (v, s) in res and (ref, s) in res]
        dtxt = f"   - {ref}: {np.mean(d):+.4f} {[round(x, 4) for x in d]}" if v != ref and d else ""
        print(f"  {v:13s} {np.mean(vals):.4f}  per seed {[round(x, 4) for x in vals]}{dtxt}")
    print(f"\nPredictions -> {out}")


if __name__ == "__main__":
    main()
