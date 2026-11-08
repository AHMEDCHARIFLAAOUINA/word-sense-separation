#!/usr/bin/env python
"""Pipeline entry point: hidden-state extraction, metrics and plots for one model and one word.

    python run.py --model qwen2.5-7b --word bank
    python run.py --all-models --word bank --stage metrics"""

import argparse
import sys
import traceback

from src.pipeline.extract_embeddings_target import run_extraction_target
from src.pipeline.extract_embeddings_final_token import run_extraction_final_token
from src.pipeline.compute_metrics import run_metrics
from src.pipeline.generate_plots import run_plots
from src.pipeline.select_layers import run_layer_selection
from src.utils.paths import get_results_root

ALL_MODELS = [
    "llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b",
    "qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b",
    "bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base",
]

STAGES = ["extract", "metrics", "plots", "select_layers", "all"]


def run_one(model, word, stage, batch_size, strict, final_token=False):
    if stage in ("extract", "all"):
        if final_token:
            run_extraction_final_token(model, word, batch_size=batch_size)
        else:
            run_extraction_target(model, word, batch_size=batch_size, strict=strict)
    if stage in ("metrics", "all"):
        run_metrics(model, word)
    if stage in ("select_layers", "all"):
        run_layer_selection(model, word)
    if stage in ("plots", "all"):
        run_plots(model, word)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", type=str)
    g.add_argument("--all-models", action="store_true")
    ap.add_argument("--word", type=str, default="bank")
    ap.add_argument("--stage", type=str, default="all", choices=STAGES)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-strict", action="store_true",
                    help="Disable extraction validation gates (not recommended)")
    ap.add_argument("--skip", type=str, default="",
                    help="Comma-separated models to skip")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--final-token", action="store_true",
                    help="Take the last token rather than locating a target word")
    args = ap.parse_args()

    strict = not args.no_strict
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    models = [m for m in ALL_MODELS if m not in skip] if args.all_models else [args.model]

    print(f"results root : {get_results_root()}")
    print(f"word         : {args.word}")
    print(f"stage        : {args.stage}")
    print(f"strict gates : {strict}")
    print(f"models       : {len(models)}")
    if not strict:
        print("WARNING: validation gates disabled")

    failed = []
    for i, m in enumerate(models, 1):
        print(f"\n{'='*70}\n[{i}/{len(models)}] {m}\n{'='*70}")
        try:
            run_one(m, args.word, args.stage, args.batch_size, strict, args.final_token)
        except Exception as e:
            failed.append((m, type(e).__name__, str(e).split("\n")[0]))
            print(f"\n!! FAILED: {m}: {type(e).__name__}: {e}")
            if not args.continue_on_error:
                traceback.print_exc()
                sys.exit(1)

    if failed:
        print(f"\n{len(failed)} model(s) failed:")
        for m, t, msg in failed:
            print(f"  {m}: {t}: {msg[:90]}")
        sys.exit(1)
    print(f"\nDone. {len(models)} model(s), word='{args.word}', stage='{args.stage}'.")


if __name__ == "__main__":
    main()
