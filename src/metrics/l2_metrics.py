"""Layer-wise L2 distances."""
import numpy as np
from scipy.spatial.distance import cdist


def compute_l2_metrics(embeddings):
    """Pairwise L2 distances between unit-normalised representations."""
    num_sent, num_layers, _ = embeddings.shape

    avg_l2 = []
    l2_matrices = []

    for l in range(num_layers):
        layer_emb = embeddings[:, l, :].astype(np.float32)

        norm = np.linalg.norm(layer_emb, axis=-1, keepdims=True)
        norm[norm == 0] = 1.0
        layer_emb = layer_emb / norm

        mat = cdist(layer_emb, layer_emb, metric="euclidean").astype(np.float32)

        l2_matrices.append(mat)

        mask = ~np.eye(num_sent, dtype=bool)
        avg_l2.append(float(mat[mask].mean()))

    return np.array(avg_l2), l2_matrices
