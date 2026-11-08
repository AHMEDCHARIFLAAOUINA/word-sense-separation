"""Line plots of the metrics across layers."""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_metric_trend(csv_path, x_col, y_col, title, ylabel, output_path):
    df = pd.read_csv(csv_path)

    plt.figure()
    plt.plot(df[x_col], df[y_col])
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_all_trend_plots(metrics_dir, trends_dir, model_name):
    metrics_dir = Path(metrics_dir)
    trends_dir = Path(trends_dir)
    trends_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_trend(
        metrics_dir / "cosine_layerwise.csv",
        "layer",
        "avg_cosine",
        f"{model_name} - Average Cosine Across Layers",
        "avg_cosine",
        trends_dir / "cosine_trend.png",
    )

    plot_metric_trend(
        metrics_dir / "l2_layerwise.csv",
        "layer",
        "avg_l2",
        f"{model_name} - Average L2 Across Layers",
        "avg_l2",
        trends_dir / "l2_trend.png",
    )

    plot_metric_trend(
        metrics_dir / "layer_drift.csv",
        "layer",
        "drift_to_next",
        f"{model_name} - Layer Drift",
        "drift_to_next",
        trends_dir / "drift_trend.png",
    )

    plot_metric_trend(
        metrics_dir / "semantic_separation.csv",
        "layer",
        "separation",
        f"{model_name} - Semantic Separation Across Layers",
        "separation",
        trends_dir / "semantic_separation_trend.png",
    )
