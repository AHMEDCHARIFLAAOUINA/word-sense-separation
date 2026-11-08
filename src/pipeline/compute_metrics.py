"""Layer-wise metrics from the saved hidden states (separation, cosine, L2, drift)."""
import json
import numpy as np

from src.metrics.cosine_metrics import compute_cosine_metrics
from src.metrics.l2_metrics import compute_l2_metrics
from src.metrics.drift_metrics import compute_layer_drift
from src.metrics.semantic_separation import compute_semantic_separation
from src.utils.paths import ensure_model_result_dirs, get_standard_result_files


def run_metrics(model_name: str, word: str = "bank"):
    print(f"[metrics] {model_name} / {word}")

    paths = ensure_model_result_dirs(model_name, word)
    files = get_standard_result_files(model_name, word)

    # Load embeddings
    data = np.load(files["embedding_tensor"])
    embeddings = data["embeddings"]

    print("Original dtype:", embeddings.dtype)
    print("Original max value:", np.max(np.abs(embeddings)))

    # cast before computing the norms
    embeddings = embeddings.astype(np.float32)

    # Normalize
    norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    embeddings = embeddings / norm

    print("Max after normalization:", np.max(np.abs(embeddings)))

    # Load labels
    with open(files["labels"], "r", encoding="utf-8") as f:
        labels = json.load(f)["labels"]

    # Compute metrics
    avg_cosine, cosine_mats = compute_cosine_metrics(embeddings)
    avg_l2, l2_mats = compute_l2_metrics(embeddings)
    drift = compute_layer_drift(embeddings)
    separation = compute_semantic_separation(embeddings, labels)

    # Cosine
    with open(files["cosine_metrics_csv"], "w") as f:
        f.write("layer,avg_cosine\n")
        for i, val in enumerate(avg_cosine):
            f.write(f"{i},{val}\n")

    # L2
    l2_path = paths["metrics"] / "l2_layerwise.csv"
    with open(l2_path, "w") as f:
        f.write("layer,avg_l2\n")
        for i, val in enumerate(avg_l2):
            f.write(f"{i},{val}\n")

    # Drift
    drift_path = paths["metrics"] / "layer_drift.csv"
    with open(drift_path, "w") as f:
        f.write("layer,drift_to_next\n")
        for i, val in enumerate(drift):
            f.write(f"{i},{val}\n")

    # Separation
    sep_path = paths["metrics"] / "semantic_separation.csv"
    with open(sep_path, "w") as f:
        f.write("layer,separation\n")
        for i, val in enumerate(separation):
            f.write(f"{i},{val}\n")

    # Matrices
    for i, mat in enumerate(cosine_mats):
        np.save(paths["metrics"] / f"cosine_matrix_layer_{i}.npy", mat)

    for i, mat in enumerate(l2_mats):
        np.save(paths["metrics"] / f"l2_matrix_layer_{i}.npy", mat)

    print("metrics saved (cosine, l2, drift, separation)")