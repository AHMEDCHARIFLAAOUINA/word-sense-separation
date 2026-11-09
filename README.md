# Downstream Effects of Controlling Word-Sense Separation in Large Language Models

Code, dataset and result tables for the paper

> Ahmed Charif Laaouina, Shuoran Jiang and Qingcai Chen.
> *Downstream Effects of Controlling Word-Sense Separation in Large Language Models.*
> Journal of Artificial Intelligence Research (under review).

The paper (1) relates layer-wise word-sense separation to the spectral geometry of the weights in
thirteen language models, using a new dataset of 1,000 ambiguous SemCor words, and (2) adds a
sense-aware *pull* loss to LoRA fine-tuning of five decoders on GSM8K and measures the downstream
effects on GSM8K and eleven other benchmarks.

## Contents

| Path | Content |
|---|---|
| `data/raw/semcor_words/` | the dataset: one JSON file per word (sentences, WordNet sense labels) and `_manifest.json` |
| `configs/words/` | one configuration file per word |
| `run.py`, `src/` | the pipeline: hidden-state extraction, separation measure, weight geometry, analysis, fine-tuning |
| `scripts/dataset/` | building and checking the dataset from SemCor 3.0 |
| `scripts/geometry_separation/` | Section 4 of the paper: batch runs, spectral truncation, robustness, tables and figures |
| `scripts/openhownet_comparison/` | Section 4.1: comparison with the OpenHowNet-based measure (run from the OpenHowNet project folder, not included) |
| `scripts/sense_pull_finetuning/` | Sections 5–6: launchers for the pull-loss runs, evaluation, summaries and tables |
| `results/tables/` | the result tables of the paper as CSV files (`geometry_separation`, `openhownet_comparison`, `sense_pull_finetuning`) |
| `results/figures/` | the figures of the paper (PDF) |
| `environment/` | conda environments: `llm_env` (all models except qwen-7b), `qwen7_env` (qwen-7b), `opencompass` (benchmarks) |

The full raw outputs (per-word separation curves for 13 models, weight geometry, truncation results,
per-item benchmark predictions; about 860 MB) are too large for this repository. They will be archived on
Zenodo when the paper is published, and the link will be added here.

## The dataset

SemCor 3.0; 1,000 ambiguous words; the two most frequent WordNet senses of each word; balanced classes;
one surface form per word; 16,046 sentences, each prefixed `"Example sentence: "`. The word *example*
has no results because it also appears in the prefix; the analyses therefore use 999 words. The
fine-tuning experiments hold out 50 words as a fixed probe and train on the other 949.

## Setup

```bash
conda env create -f environment/llm_env.yml        # transformers 5.4.0, peft 0.20.0, torch 2.7.1
conda activate llm_env
```

Paths are set with environment variables (defaults in brackets):

| Variable | Meaning |
|---|---|
| `WSS_MODELS` [`models`] | folder with the pretrained models (e.g. `Llama/Llama-3-8b`) |
| `WSS_BENCH_DATA` [`benchmark_data`] | folder with the benchmark data used for GSM8K and the eleven benchmarks |
| `WSS_PIPELINE` [`.`] | root of this repository (contains `data/raw/semcor_words`) |

Models are not included; they are available from their original providers on Hugging Face.

## Reproducing the paper

| Paper | Script |
|---|---|
| Separation curves (Section 4.1) | `python run.py --all-models --word <word>`, or `scripts/geometry_separation/batch_run_words.py` for all words |
| Weight geometry vs separation (Section 4.2) | `src/analysis/compute_geometry.py`, `src/analysis/cross_word.py`, `scripts/geometry_separation/semcor_robustness.py` |
| Spectral truncation (Section 4.3) | `scripts/geometry_separation/batch_intervention.py`, `intervention_semcor_analysis.py` |
| Pull-loss fine-tuning (Sections 5–6) | `scripts/sense_pull_finetuning/run_llama_full.py`, `run_qwen_full.py`, `run_pull_layers.py`, `run_llama_10seeds.py` |
| Benchmarks and tables (Section 6) | `scripts/sense_pull_finetuning/eval_final.py`, `eval_broad.py`, `make_finetuning_tables.py` |

The pull loss is implemented in `src/finetune/sense_pull_regulariser.py` and selected with `--loss ta_pull`;
`ta_pull` is its internal name and appears in the names of the result files. The reference runs use
`--baseline`. Other loss variants in the code (`--loss ta`, `pushpull`) belong to earlier experiments and
are not used in the paper.

All settings (LoRA rank 16, α 32, learning rate 1e-4, λ = 1, 1,991 steps, seeds) are listed in
Appendix A of the paper.

## Licence

Code: MIT (`LICENSE`). Dataset and results: CC BY 4.0, with the SemCor/WordNet notice (`DATA_LICENSE.md`).

## Citation

```bibtex
@article{laaouina2026downstream,
  title   = {Downstream Effects of Controlling Word-Sense Separation in Large Language Models},
  author  = {Laaouina, Ahmed Charif and Jiang, Shuoran and Chen, Qingcai},
  journal = {Journal of Artificial Intelligence Research},
  year    = {2026},
  note    = {Under review}
}
```