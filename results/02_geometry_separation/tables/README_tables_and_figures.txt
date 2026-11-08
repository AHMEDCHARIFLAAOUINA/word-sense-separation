results/final_semcor/ - the 1,000-word SemCor study
============================================================
assembled 2026-09-26 07:02 by scripts/semcor/assemble_final_semcor.py

Dataset: 1,000 words from SemCor 3.0, two WordNet senses each, balanced
classes, one surface form per word, 16,046 sentences.

Headline (q_proj stable rank against post-block Sep(l)):
  decoders   linear     mean +0.323  9/9 positive  Wilcoxon p 0.0039
  decoders   quadratic  mean +0.179  8/9 positive  Wilcoxon p 0.0117
  llama      linear     mean +0.352  4/4 positive  Wilcoxon p 0.1250
  llama      quadratic  mean +0.267  4/4 positive  Wilcoxon p 0.1250
  qwen       linear     mean +0.300  5/5 positive  Wilcoxon p 0.0625
  qwen       quadratic  mean +0.109  4/5 positive  Wilcoxon p 0.1875
  encoders   linear     mean +0.088  3/4 positive  Wilcoxon p 0.3750
  encoders   quadratic  mean -0.058  1/4 positive  Wilcoxon p 0.2500

Intervention, truncation to 80% of stable rank:
  q_proj   peak retention  +0.0168  7/9 up  Wilcoxon p 0.0273
  q_proj   mean retention  +0.0005  6/9 up  Wilcoxon p 0.4961
  v_proj   peak retention  -0.1779  3/9 up  Wilcoxon p 0.0547
  v_proj   mean retention  -0.1579  3/9 up  Wilcoxon p 0.2031

Files
------------------------------------------------------------
  fig1_separation_by_architecture.pdf    Sep(l) against relative depth, median per family with the interquartile band
  fig1_separation_by_architecture.png    Sep(l) against relative depth, median per family with the interquartile band
  fig2_partial_r_distribution.pdf        distribution of the per-word partial r, per model
  fig2_partial_r_distribution.png        distribution of the per-word partial r, per model
  fig3_model_consistency.pdf             per-model mean partial r with bootstrap intervals
  fig3_model_consistency.png             per-model mean partial r with bootstrap intervals
  fig4_measure_comparison.pdf            q_proj stable rank against up_proj effective rank
  fig4_measure_comparison.png            q_proj stable rank against up_proj effective rank
  fig5_intervention_distribution.pdf     peak retention over 1,000 words, per model
  fig5_intervention_distribution.png     peak retention over 1,000 words, per model
  fig6_intervention_model_means.pdf      mean peak retention - 1 with bootstrap intervals
  fig6_intervention_model_means.png      mean peak retention - 1 with bootstrap intervals
  fig7_context_baseline.pdf              context-only against target-only accuracy, per word
  fig7_context_baseline.png              context-only against target-only accuracy, per word
  table1_dataset_properties.csv          dataset properties, medians over the 1,000 words
  table1_per_word.csv                    the same per word: Jaccard, length, context-only, target-only, Sep max, peak depth
  table2_by_model.csv                    q_proj stable rank against Sep(l), per model, mean over words
  table3_model_level.csv                 model-level tests by group and depth control
  table3b_bootstrap_models.csv           per-model bootstrap over words, with the weighted mean
  table3c_bootstrap_groups.csv           group-level bootstrap, one value per model
  table3d_up_proj_models.csv             comparison measure: up_proj effective rank, per model
  table3e_up_proj_summary.csv            comparison measure, model-level tests
  table5b_truncate_models.csv            intervention: retention per model, q_proj and v_proj
  table5c_model_level.csv                intervention: model-level tests on (retention - 1)
  table6_context_baseline.csv            context-only, target-only and combined accuracy per word
  tableS3_cue_position.csv               cue-position proxy, median split (inconclusive)
  tableS3b_cue_position_quartiles.csv    the same with a quartile split (inconclusive)

