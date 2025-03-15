import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


def visualize_layer_activations(activations, label, reduction_type='mean', layer_indexes=0):
    L, B, N, H = activations.shape
    assert reduction_type in ["mean", "first", "sum"], "Invalid reduction type. Choose from ['mean', 'first', 'sum']."
    if reduction_type == "mean":
        reduced_b_activations = activations.mean(dim=1)  # Shape: (L, N, H)
    elif reduction_type == "sum":
        reduced_b_activations = activations.sum(dim=1)  # Shape: (L, N, H)
    elif reduction_type == "first":
        reduced_b_activations = activations[:,0,:,:]  # First batch element (L, N, H)
    if isinstance(reduced_b_activations, torch.Tensor):
        layer_activations = reduced_b_activations.detach().cpu().numpy()

    for layer_idx in layer_indexes:
        plt.figure(figsize=(12, 6))
        sns.heatmap(reduced_b_activations[layer_idx], cmap="coolwarm", xticklabels=50, yticklabels=10)
        plt.xlabel("Hidden Size (768)")
        plt.ylabel("Token Position (128)")
        plt.title(f"Activations for Layer {layer_idx} (Label: {label})")
        plt.show()

def visualize_c_heatmap(c, sentiment,save_path=None):
    plt.figure(figsize=(12, 6))
    sns.heatmap(c, cmap="Reds", cbar=True, linewidths=0.5)

    plt.xlabel("Dictionary Features (2d)")
    plt.ylabel("Tokens (N)")
    plt.title(f"Sparse Encoding Heatmap (c Matrix), real sentiment: {sentiment}")

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Heatmap saved to {save_path}")
    plt.show()
