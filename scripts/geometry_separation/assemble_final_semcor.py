"""Collect the final tables and figures into results/final_semcor/ with a README.

    python scripts/geometry_separation/assemble_final_semcor.py"""
import os, shutil, sys, pathlib
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

SRC_SEMFINAL = "results/analysis/_semcor_final"
SRC_CROSS = "results/analysis/_cross_word"
SRC_CTX = "results/analysis/_context_baseline"
OUT = "results/final_semcor"

# destination name  <-  source path
FILES = {
    # tables
    "table1_dataset_properties.csv": f"{SRC_SEMFINAL}/table1_semcor_summary.csv",
    "table1_per_word.csv":           f"{SRC_SEMFINAL}/table1_semcor_per_word.csv",
    "table2_by_model.csv":           f"{SRC_CROSS}/cross_word_q_proj_stable_rank_semcor_models.csv",
    "table3_model_level.csv":        f"{SRC_CROSS}/cross_word_q_proj_stable_rank_semcor_summary.csv",
    "table3b_bootstrap_models.csv":  f"{SRC_CROSS}/cross_word_q_proj_stable_rank_semcor_robustness_models.csv",
    "table3c_bootstrap_groups.csv":  f"{SRC_CROSS}/cross_word_q_proj_stable_rank_semcor_robustness_groups.csv",
    "table3d_up_proj_models.csv":    f"{SRC_CROSS}/cross_word_up_proj_erank_semcor_models.csv",
    "table3e_up_proj_summary.csv":   f"{SRC_CROSS}/cross_word_up_proj_erank_semcor_summary.csv",
    "table5b_truncate_models.csv":   f"{SRC_SEMFINAL}/intervention_semcor_models.csv",
    "table5c_model_level.csv":       f"{SRC_SEMFINAL}/intervention_semcor_model_level.csv",
    "table6_context_baseline.csv":   f"{SRC_CTX}/context_baseline_semcor.csv",
    "tableS3_cue_position.csv":      f"{SRC_CROSS}/cue_position_semcor.csv",
    "tableS3b_cue_position_quartiles.csv": f"{SRC_CROSS}/cue_position_semcor_quartiles.csv",
    # figures
    "fig1_separation_by_architecture.pdf": f"{SRC_SEMFINAL}/figS_semcor4_separation_by_architecture.pdf",
    "fig1_separation_by_architecture.png": f"{SRC_SEMFINAL}/figS_semcor4_separation_by_architecture.png",
    "fig2_partial_r_distribution.pdf":     f"{SRC_SEMFINAL}/figS_semcor1_distribution.pdf",
    "fig2_partial_r_distribution.png":     f"{SRC_SEMFINAL}/figS_semcor1_distribution.png",
    "fig3_model_consistency.pdf":          f"{SRC_SEMFINAL}/figS_semcor2_model_means.pdf",
    "fig3_model_consistency.png":          f"{SRC_SEMFINAL}/figS_semcor2_model_means.png",
    "fig4_measure_comparison.pdf":         f"{SRC_SEMFINAL}/figS_semcor5_measure_comparison.pdf",
    "fig4_measure_comparison.png":         f"{SRC_SEMFINAL}/figS_semcor5_measure_comparison.png",
    "fig5_intervention_distribution.pdf":  f"{SRC_SEMFINAL}/figS_semcor6_intervention_distribution.pdf",
    "fig5_intervention_distribution.png":  f"{SRC_SEMFINAL}/figS_semcor6_intervention_distribution.png",
    "fig6_intervention_model_means.pdf":   f"{SRC_SEMFINAL}/figS_semcor7_intervention_model_means.pdf",
    "fig6_intervention_model_means.png":   f"{SRC_SEMFINAL}/figS_semcor7_intervention_model_means.png",
    "fig7_context_baseline.pdf":           f"{SRC_SEMFINAL}/figS_semcor3_context_baseline.pdf",
    "fig7_context_baseline.png":           f"{SRC_SEMFINAL}/figS_semcor3_context_baseline.png",
}

DESC = {
    "table1_dataset_properties.csv": "dataset properties, medians over the 1,000 words",
    "table1_per_word.csv": "the same per word: Jaccard, length, context-only, target-only, Sep max, peak depth",
    "table2_by_model.csv": "q_proj stable rank against Sep(l), per model, mean over words",
    "table3_model_level.csv": "model-level tests by group and depth control",
    "table3b_bootstrap_models.csv": "per-model bootstrap over words, with the weighted mean",
    "table3c_bootstrap_groups.csv": "group-level bootstrap, one value per model",
    "table3d_up_proj_models.csv": "comparison measure: up_proj effective rank, per model",
    "table3e_up_proj_summary.csv": "comparison measure, model-level tests",
    "table5b_truncate_models.csv": "intervention: retention per model, q_proj and v_proj",
    "table5c_model_level.csv": "intervention: model-level tests on (retention - 1)",
    "table6_context_baseline.csv": "context-only, target-only and combined accuracy per word",
    "tableS3_cue_position.csv": "cue-position proxy, median split (inconclusive)",
    "tableS3b_cue_position_quartiles.csv": "the same with a quartile split (inconclusive)",
    "fig1_separation_by_architecture": "Sep(l) against relative depth, median per family with the interquartile band",
    "fig2_partial_r_distribution": "distribution of the per-word partial r, per model",
    "fig3_model_consistency": "per-model mean partial r with bootstrap intervals",
    "fig4_measure_comparison": "q_proj stable rank against up_proj effective rank",
    "fig5_intervention_distribution": "peak retention over 1,000 words, per model",
    "fig6_intervention_model_means": "mean peak retention - 1 with bootstrap intervals",
    "fig7_context_baseline": "context-only against target-only accuracy, per word",
}


def readme(copied, missing):
    L = []
    A = L.append
    A("results/final_semcor/ - the 1,000-word SemCor study")
    A("=" * 60)
    A(f"assembled {datetime.now():%Y-%m-%d %H:%M} by scripts/geometry_separation/assemble_final_semcor.py")
    A("")
    A("Dataset: 1,000 words from SemCor 3.0, two WordNet senses each, balanced")
    A("classes, one surface form per word, 16,046 sentences.")
    A("")
    try:
        g = pd.read_csv(f"{OUT}/table3c_bootstrap_groups.csv")
        A("Headline (q_proj stable rank against post-block Sep(l)):")
        for _, r in g.iterrows():
            A(f"  {r.group:10s} {r.control:10s} mean {r['mean']:+.3f}  "
              f"{int(r.n_positive)}/{int(r.n_models)} positive  "
              f"Wilcoxon p {r.wilcoxon_p:.4f}")
        A("")
    except Exception:
        pass
    try:
        m = pd.read_csv(f"{OUT}/table5c_model_level.csv")
        A("Intervention, truncation to 80% of stable rank:")
        for _, r in m.iterrows():
            A(f"  {r.projection:8s} {r.measure:15s} {r.mean_change:+.4f}  "
              f"{int(r.n_positive)}/{int(r.n_models)} up  "
              f"Wilcoxon p {r.wilcoxon_p:.4f}")
        A("")
    except Exception:
        pass
    A("Files")
    A("-" * 60)
    for n in sorted(copied):
        key = n.rsplit(".", 1)[0] if n.endswith((".pdf", ".png")) else n
        d = DESC.get(key, DESC.get(n, ""))
        A(f"  {n:38s} {d}")
    if missing:
        A("")
        A("Not present (the source was not found):")
        for n in sorted(missing):
            A(f"  {n}")
    return "\n".join(L)


def main():
    os.makedirs(OUT, exist_ok=True)
    copied, missing = [], []
    for dst, src in FILES.items():
        if os.path.exists(src):
            shutil.copy2(src, f"{OUT}/{dst}")
            copied.append(dst)
        else:
            missing.append(f"{dst}  <- {src}")
    with open(f"{OUT}/README.txt", "w") as f:
        f.write(readme(copied, missing))
    print(f"{len(copied)} files -> {OUT}")
    for n in sorted(copied):
        print("  ", n)
    if missing:
        print("\nmissing sources:")
        for n in missing:
            print("  ", n)


if __name__ == "__main__":
    main()
