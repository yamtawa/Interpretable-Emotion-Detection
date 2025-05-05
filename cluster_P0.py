import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap
import os

def cluster_and_plot(data_dict,saved_name, perplexity=60, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)

    all_data = []
    all_labels = []

    for cls, tensor in data_dict.items():
        n = tensor.shape[0]
        flattened = tensor.view(n, -1)
        all_data.append(flattened)
        all_labels.extend([cls] * n)

    all_data = torch.cat(all_data, dim=0).cpu().numpy()
    all_labels = np.array(all_labels)

    # Encode string labels into integers
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(all_labels)

    # t-SNE for 2D projection
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_embedded = tsne.fit_transform(all_data)

    # Define more distinguishable colors
    distinct_colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231",
        "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#fabebe",
        "#008080", "#e6beff", "#aa6e28", "#fffac8", "#800000",
        "#aaffc3", "#808000", "#ffd8b1", "#000080", "#808080",
        "#FFFFFF", "#000000"  # add more if needed
    ]
    cmap = ListedColormap(distinct_colors[:len(np.unique(numeric_labels))])

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_embedded[:, 0], X_embedded[:, 1],
        c=numeric_labels, cmap=cmap, alpha=0.8,
        edgecolors='k', linewidth=0.3
    )
    plt.title("t-SNE projection (colored by true labels)")
    plt.grid(True)

    # Use scatter to generate legend elements
    handles, _ = scatter.legend_elements()
    plt.legend(handles=handles, labels=list(label_encoder.classes_), title="Classes", loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'tsne_projection_{saved_name}.png'), bbox_inches='tight')
    plt.close()
