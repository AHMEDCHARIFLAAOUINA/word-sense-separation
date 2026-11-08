"""PCA scatter plots of the target-token representations, one per layer."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def generate_pca_plots(embeddings, labels, output_dir):
    num_layers = embeddings.shape[1]

    senses = sorted(set(labels))
    label_colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(senses)}
    colors = [label_colors[l] for l in labels]

    for layer in range(num_layers):
        layer_emb = embeddings[:, layer, :]

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(layer_emb)

        fig, ax = plt.subplots()
        for s in senses:
            idx = [i for i, l in enumerate(labels) if l == s]
            ax.scatter(reduced[idx, 0], reduced[idx, 1],
                       color=label_colors[s], s=80, label=s)

        ax.set_title(f"PCA Layer {layer}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(f"{output_dir}/pca_layer_{layer}.png")
        plt.close(fig)
