"""Layer-wise cosine similarity between sentence representations."""
import numpy as np


def compute_cosine_metrics(embeddings):
    """embeddings: [num_sentences, num_layers, hidden_size]"""

    num_sent, num_layers, dim = embeddings.shape

    avg_cosine = []
    cosine_matrices = []

    for l in range(num_layers):
        layer_emb = embeddings[:, l, :]

        layer_emb = layer_emb.astype(np.float32)

        # normalise (the inputs may not be unit length)
        norm = np.linalg.norm(layer_emb, axis=-1, keepdims=True)
        norm[norm == 0] = 1.0
        layer_emb = layer_emb / norm

        # cosine similarity = dot product (since normalized)
        sim_matrix = np.dot(layer_emb, layer_emb.T)

        sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

        cosine_matrices.append(sim_matrix)

        # remove diagonal
        mask = ~np.eye(num_sent, dtype=bool)
        avg = sim_matrix[mask].mean()
        avg_cosine.append(float(avg))

    return np.array(avg_cosine), cosine_matrices