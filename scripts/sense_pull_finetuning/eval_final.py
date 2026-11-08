"""Generation test of one model on one dataset (GSM8K uses the stop rule: answer before a new 'Question').

    python scripts/sense_pull_finetuning/eval_final.py --eval gsm8k_full --model qwen2-7b --adapter <dir>"""
import argparse, csv, random, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_accuracy as ga   # noqa: E402  (generate, gsm8k_correct, trivia_correct, ft)
import ood_accuracy as oa   # noqa: E402  (load_math, load_nq)


def load(ev, seed):
    if ev == "math":
        return oa.load_math(0)
    if ev == "nq":
        return oa.load_nq(0)
    _, test = ga.ft.TASKS[ev](random.Random(seed))
    return [{"prompt": r["prompt"], "gold": [r["final"]]} for r in test]


STOP_MARKERS = ("[Question]", "\nQuestion:", "\n\nQuestion")


def cut_first_answer(pred):
    """Keep only the first answer: cut where the model starts a new question."""
    cuts = [pred.find(m) for m in STOP_MARKERS if pred.find(m) > 0]
    return pred[:min(cuts)] if cuts else pred


def correct(ev, pred, golds):
    if ev in ("math", "gsm8k", "gsm8k_full"):
        return ga.gsm8k_correct(pred, golds[0])
    return any(ga.trivia_correct(pred, g) for g in golds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, choices=["gsm8k", "gsm8k_full", "triviaqa", "math", "nq"])
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--seed", type=int, default=0, help="test split (gsm8k / triviaqa only)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N items only")
    ap.add_argument("--model", default="llama-3-8b", help="base model of the adapter")
    ap.add_argument("--out-dir", default="final_eval",
                    help="folder under results/analysis/_finetune (final_eval_v2 = stop-rule re-evaluation)")
    a = ap.parse_args()

    from peft import PeftModel
    from src.models.hf_loader import load_model_and_tokenizer

    test = load(a.eval, a.seed)
    if a.limit:
        test = test[:a.limit]
    tok, model = load_model_and_tokenizer(a.model)[:2]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if a.adapter == "none":
        name = ("pretrained" + ("" if a.model == "llama-3-8b" else f"_{a.model}")
                + (f"_s{a.seed}" if a.eval in ("gsm8k", "triviaqa") else ""))
    else:
        model = PeftModel.from_pretrained(model, a.adapter)
        name = Path(a.adapter).name
    model.eval()
    device = next(model.parameters()).device

    preds = ga.generate(model, tok, [r["prompt"] for r in test], device,
                        256 if a.eval in ("math", "gsm8k", "gsm8k_full") else 24)
    ok = [correct(a.eval, pr, r["gold"]) for pr, r in zip(preds, test)]
    ok_stop = [correct(a.eval, cut_first_answer(pr), r["gold"]) for pr, r in zip(preds, test)]

    out = ga.ROOT / "results/analysis/_finetune" / a.out_dir / f"{a.eval}__{name}{'_limit' if a.limit else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "gold", "pred", "correct", "correct_stop"])
        w.writeheader()
        for i, (pr, r, c, cs) in enumerate(zip(preds, test, ok, ok_stop)):
            w.writerow({"item": i, "gold": " | ".join(r["gold"]),
                        "pred": pr[:300] if a.out_dir == "final_eval" else pr,
                        "correct": int(c), "correct_stop": int(cs)})
    print(f"{name} on {a.eval}: accuracy {np.mean(ok):.4f} ({sum(ok)}/{len(ok)})  stop-rule accuracy {np.mean(ok_stop):.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
