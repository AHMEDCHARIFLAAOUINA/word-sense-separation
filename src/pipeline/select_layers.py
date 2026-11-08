"""Rank the layers by stability and keep the top four."""
import json

from src.utils.paths import ensure_model_result_dirs
from src.metrics.layer_ranking import rank_layers


def run_layer_selection(model_name: str, word: str = "bank"):

    paths = ensure_model_result_dirs(model_name, word)

    cosine = paths["metrics"] / "cosine_layerwise.csv"
    l2 = paths["metrics"] / "l2_layerwise.csv"
    drift = paths["metrics"] / "layer_drift.csv"

    ranking = rank_layers(cosine, l2, drift)

    ranking.to_csv(paths["summaries"] / "layer_ranking.csv", index=False)

    top_layers = ranking.head(4)["layer"].tolist()

    with open(paths["summaries"] / "top_layers.json", "w") as f:
        json.dump({"top_layers": top_layers}, f, indent=2)

    print("Top layers:", top_layers)
