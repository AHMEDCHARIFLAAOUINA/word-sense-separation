"""Benchmark scores (C-Eval, CMMLU, GSM8K) of the 13 models next to their peak separation.
Reads the OpenCompass archives and the per-word separation files in paper_final.

    cd paper_final; python code/scripts/geometry_separation/make_benchmark_table.py"""
import argparse, csv, io, json, re, tarfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".", help="paper_final folder")
a = ap.parse_args()
PART = Path(a.root) / "02_geometry_separation"
SCORES, SEP, OUT = PART / "data/benchmark_scores", PART / "data/separation_per_word", PART / "tables"

# model: (family folder, archive prefix, number of blocks)
MODELS = {
    "llama-7b": ("llama", "llama7b", 32), "llama-2-7b": ("llama", "llama2_7b", 32),
    "llama-3-8b": ("llama", "llama3_8b", 32), "llama-3.1-8b": ("llama", "llama3_1_8b", 32),
    "qwen-7b": ("qwen", "qwen_7b", 32), "qwen1.5-7b": ("qwen", "qwen1_5_7b", 32),
    "qwen2-7b": ("qwen", "qwen2_7b", 28), "qwen2.5-7b": ("qwen", "qwen2_5_7b", 28),
    "qwen3-8b": ("qwen", "qwen3_8b", 36),
    "bert-base": ("bert", "bert_base", 12), "roberta-base": ("bert", "roberta_base", 12),
    "spanbert-base-cased": ("bert", "spanbert_base", 12), "xlm-roberta-base": ("bert", "xlm_roberta_base", 12),
}
FAMILY = {"llama": "LLaMA", "qwen": "Qwen", "bert": "BERT-family"}
BENCH = ("ceval", "cmmlu", "gsm8k")
# the archived output of this run (qwen_7b_gsm8k.tar.gz) is empty;
# this is the score reported from that run, marked with * in the table
NO_ARCHIVE = {("qwen_7b", "gsm8k"): 54.13}


def score(prefix, bench):
    with tarfile.open(SCORES / f"{prefix}_{bench}.tar.gz") as t:
        names = t.getnames()
        summ = sorted(n for n in names if re.search(r"summary/summary_.*\.csv$", n))
        if summ:
            rows = list(csv.reader(io.TextIOWrapper(t.extractfile(summ[-1]), "utf-8")))[1:]
            vals = {r[0]: float(r[-1]) for r in rows if r and r[-1] not in ("-", "")}
            if bench in vals:
                return vals[bench]
            sub = [v for k, v in vals.items() if k.startswith(bench + "-")]
            return float(np.mean(sub)) if sub else vals[next(iter(vals))]
        for n in names:
            if n.endswith(".json"):
                d = json.load(t.extractfile(n))
                if isinstance(d, dict) and "accuracy" in d:
                    acc = d["accuracy"]
                    return acc * 100 if acc <= 1 else acc
                items = next((v for v in (d.values() if isinstance(d, dict) else [d]) if isinstance(v, list)), None)
                if items and isinstance(items[0], dict) and "correct" in items[0]:
                    return 100 * float(np.mean([bool(x["correct"]) for x in items]))
    if (prefix, bench) in NO_ARCHIVE:
        return None
    raise ValueError(f"no score found in {prefix}_{bench}.tar.gz")


def curves(model, fam):
    out = {}
    for f in SEP.glob(f"*/{fam}/{model}/metrics/semantic_separation.csv"):
        rows = list(csv.DictReader(open(f)))
        out[f.parts[-5]] = np.array([float(r["separation"]) for r in rows])
    return out


all_curves = {m: curves(m, fam) for m, (fam, _, _) in MODELS.items()}
common = sorted(set.intersection(*(set(c) for c in all_curves.values())))

rows, out_csv = [], []
for m, (fam, prefix, n_blocks) in MODELS.items():
    s = [score(prefix, b) for b in BENCH]
    marks = ["*" if x is None else "" for x in s]
    s = [NO_ARCHIVE[(prefix, b)] if x is None else x for x, b in zip(s, BENCH)]
    mean_curve = np.mean([all_curves[m][w] for w in common], axis=0)
    peak, layer = float(mean_curve.max()), int(mean_curve.argmax())
    rows.append([FAMILY[fam], m] + [f"{x:.2f}{mk}" for x, mk in zip(s, marks)] + [f"{peak:.3f}", f"{layer}/{n_blocks}"])
    out_csv.append([FAMILY[fam], m] + [round(x, 2) for x in s] + [round(peak, 4), layer, n_blocks])
rows.append(["", "random baseline (4 choices)", "25.00", "25.00", "25.00", "", ""])

with open(OUT / "benchmark_13models_semcor.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["family", "model", "ceval", "cmmlu", "gsm8k", "peak_sep", "peak_hidden_state", "n_blocks"])
    w.writerows(out_csv)

shade = {"LLaMA": "#eef4fb", "Qwen": "#fbf1ef", "BERT-family": "#eef7ee", "": "#f4f4f4"}
fig, ax = plt.subplots(figsize=(12, 6.4))
ax.axis("off")
tb = ax.table(cellText=rows, cellLoc="center", loc="upper center", colWidths=[0.13, 0.25, 0.09, 0.09, 0.1, 0.17, 0.1],
              colLabels=["Family", "Model", "C-Eval", "CMMLU", "GSM8K", f"Peak Sep\n({len(common)} SemCor words)", "Peak layer"])
tb.auto_set_font_size(False)
tb.set_fontsize(11)
tb.scale(1, 1.9)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#bbbbbb")
    if r == 0:
        cell.set_facecolor("#e8e8e8")
        cell.set_text_props(weight="bold")
        cell.set_height(cell.get_height() * 1.3)
    else:
        cell.set_facecolor(shade[rows[r - 1][0]])
ax.set_title(f"Benchmark scores (accuracy, %) and peak semantic separation over the SemCor words, 13 models",
             fontsize=13, weight="bold", loc="left")
fig.text(0.125, 0.05,
         "Scores from OpenCompass. BERT-family: C-Eval / CMMLU by candidate scoring; GSM8K by 4-choice scoring, "
         "not comparable with decoder GSM8K.\n"
         "* qwen-7b GSM8K: the OpenCompass output of this run was not kept; value as reported from the run.\n"
         f"Peak Sep = maximum of the layer-wise separation averaged over the {len(common)} SemCor words common to all "
         "13 models (target token); peak layer = hidden state / number of blocks.",
         fontsize=9, va="top", color="#444")
fig.savefig(OUT / "benchmark_13models_semcor.png", dpi=200, bbox_inches="tight", facecolor="white")
print(f"{len(common)} common words; written {OUT}/benchmark_13models_semcor.png/.csv")
for r in rows[:-1]:
    print(f"  {r[1]:20s} {r[2]:>8s} {r[3]:>8s} {r[4]:>8s}   peak {r[5]}  layer {r[6]}")
