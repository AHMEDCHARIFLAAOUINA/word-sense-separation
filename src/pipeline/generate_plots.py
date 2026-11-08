"""Plots of the layer-wise metrics."""
import json
import numpy as np

from src.utils.paths import ensure_model_result_dirs, get_standard_result_files
from src.visualization.pca_plots import generate_pca_plots
from src.visualization.heatmaps import generate_heatmaps
from src.visualization.trend_plots import generate_all_trend_plots
from src.metrics.cosine_metrics import compute_cosine_metrics
from src.metrics.l2_metrics import compute_l2_metrics


def run_plots(model_name: str, word: str = "bank"):
    paths = ensure_model_result_dirs(model_name, word)
    files = get_standard_result_files(model_name, word)

    data = np.load(files["embedding_tensor"])
    embeddings = data["embeddings"]

    with open(files["labels"], "r", encoding="utf-8") as f:
        label_data = json.load(f)
    labels = label_data["labels"]

    generate_pca_plots(
        embeddings,
        labels,
        paths["pca"]
    )

    _, cosine_matrices = compute_cosine_metrics(embeddings)
    generate_heatmaps(cosine_matrices, paths["heatmaps"], prefix="cosine")

    _, l2_matrices = compute_l2_metrics(embeddings)
    generate_heatmaps(l2_matrices, paths["heatmaps"], prefix="l2")

    generate_all_trend_plots(
        paths["metrics"],
        paths["trends"],
        model_name,
    )

    print("PCA + heatmaps + trend plots generated")