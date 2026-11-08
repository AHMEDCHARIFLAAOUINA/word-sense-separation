"""Heatmaps of the layer-wise metrics."""
import matplotlib.pyplot as plt


def generate_heatmaps(matrices, output_dir, prefix="cosine"):
    for layer, mat in enumerate(matrices):
        plt.figure()
        plt.imshow(mat)
        plt.colorbar()
        plt.title(f"{prefix.capitalize()} Heatmap Layer {layer}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{prefix}_heatmap_layer_{layer}.png")
        plt.close()
