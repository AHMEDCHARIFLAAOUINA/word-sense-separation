"""Change of the representations from one layer to the next."""
import numpy as np


def compute_layer_drift(embeddings):
    """embeddings: [num_sentences, num_layers, hidden_size]"""

    num_layers = embeddings.shape[1]

    drift = []

    for l in range(num_layers - 1):
        a = embeddings[:, l, :]
        b = embeddings[:, l + 1, :]

        diff = np.linalg.norm(a - b, axis=1)

        drift.append(diff.mean())

    drift = np.array(drift)

    return drift
