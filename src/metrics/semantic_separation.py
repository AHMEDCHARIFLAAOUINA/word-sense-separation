"""Sep(l) = mean cosine(same sense) - mean cosine(different sense)."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_semantic_separation(embeddings, labels):
    """embeddings shape: [num_sentences, num_layers, hidden_size]"""

    num_layers = embeddings.shape[1]

    layers_scores = []

    labels = np.array(labels)

    for layer in range(num_layers):

        layer_emb = embeddings[:, layer, :]

        sim = cosine_similarity(layer_emb)

        intra = []
        inter = []

        for i in range(len(labels)):
            for j in range(len(labels)):

                if i == j:
                    continue

                if labels[i] == labels[j]:
                    intra.append(sim[i, j])
                else:
                    inter.append(sim[i, j])

        intra_mean = np.mean(intra)
        inter_mean = np.mean(inter)

        separation = intra_mean - inter_mean

        layers_scores.append(separation)

    return np.array(layers_scores)
