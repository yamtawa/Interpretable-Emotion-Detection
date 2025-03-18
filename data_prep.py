from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from datasets import load_dataset
from transformers import BertTokenizer
from utils import get_wanted_label
import os

class EmotionDataset(Dataset):
    def __init__(self, dataset_name='go_emotions', split='train', max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length
        self.dataset_name = dataset_name

        if dataset_name == "go_emotions":
            self.dataset = load_dataset("go_emotions", split=split)
            self.num_labels = 27  # Multi-label classification
        elif dataset_name == "dair-ai/emotion":
            self.dataset = load_dataset("dair-ai/emotion", split=split)
            self.label_map = {"anger": 0, "fear": 1, "joy": 2, "love": 3, "sadness": 4,
                              "surprise": 5}  # Single-label classification
        elif dataset_name == "jeffnyman/emotions":
            self.dataset = load_dataset("jeffnyman/emotions", split=split)
            self.label_map = {"anger": 0, "fear": 1, "joy": 2, "love": 3, "sadness": 4,
                              "surprise": 5}  # Single-label classification
        else:
            raise ValueError("Unsupported dataset")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length,
                                return_tensors='pt')
        if self.dataset_name == "go_emotions":
            labels = self.dataset[idx]["labels"]
            label_tensor = torch.zeros(self.num_labels, dtype=torch.float)
            label_tensor[labels] = 1
        else:
            label = self.dataset[idx]["label"]
            if isinstance(label, str):
                label = self.label_map[label]
            label_tensor = torch.tensor(label, dtype=torch.long)
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": label_tensor
        }


def get_dataloader(dataset_name='go_emotions', batch_size=16, split='train',ratio_small=2,labels_dl=False):
    torch.manual_seed(55) # prevents data leakage
    dataset = EmotionDataset(dataset_name=dataset_name, split=split)
    small_size = len(dataset) // ratio_small
    large_size = len(dataset) - small_size
    large_dataset, small_dataset = random_split(dataset, [large_size, small_size])
    if not labels_dl:
        large_loader = DataLoader(large_dataset, batch_size=batch_size, shuffle=False)
        small_loader = DataLoader(small_dataset, batch_size=batch_size, shuffle=False)
        return large_loader, small_loader

    # large_label_dataloaders = build_label_dataloaders(large_dataset, batch_size)
    small_label_dataloaders = build_label_dataloaders(small_dataset, batch_size)
    return None, small_label_dataloaders

def get_neuron_dataloader(wanted_labels=['anger','fear'], layer_index=0, data_type='activations',
                          batch_size=16, train_ratio=0.85, test_ratio=0.05, shuffle=True, N=128):
    torch.manual_seed(55) # prevents data leakage
    dataset = LayerDataset(data_dir="data", wanted_labels=wanted_labels, layer_index=layer_index,
                            data_type=data_type)
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = int(total_size * test_ratio)
    val_size = total_size - train_size - test_size  # Remaining data for validation

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    def convert_to_row_dataset(layer_dataset):
        """ Converts a dataset where each sample is (N x d) into independent row samples. """
        row_samples = []
        row_labels = []
        for layer_activations, label, layer_idx in layer_dataset:
            # layer_activations: Shape (N, d) → Split into N separate rows
            for row in layer_activations:
                row_samples.append(row.unsqueeze(0))  # Keep batch dimension
                row_labels.append(label)  # Keep the same label for all rows
        # Convert lists to tensors
        row_samples = torch.cat(row_samples, dim=0)  # Shape (N_total, d)
        row_labels = torch.tensor(row_labels)  # Shape (N_total,)
        return TensorDataset(row_samples, row_labels)

    train_row_dataset = convert_to_row_dataset(train_dataset)
    val_row_dataset = convert_to_row_dataset(val_dataset)
    test_row_dataset = convert_to_row_dataset(test_dataset)

    # Shuffle the train and validation row datasets
    train_loader = DataLoader(train_row_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_row_dataset, batch_size=batch_size, shuffle=False)

    # Each batch is the sorted layer rows - each layer size N
    test_loader = DataLoader(test_row_dataset, batch_size=N, shuffle=False)

    print(f"✅ Train size: {len(train_loader)}, Validation size: {len(val_loader)}, Test size: {len(test_loader)}")

    return train_loader, val_loader, test_loader

def build_label_dataloaders(subset, batch_size):
        label_to_samples = defaultdict(list)
        # Group samples by label.
        for idx in range(len(subset)):
            sample = subset[idx]
            # Ensure we use a Python int as the key.
            label = sample["labels"].item() if isinstance(sample["labels"], torch.Tensor) else sample["labels"]
            label_to_samples[label].append(sample)

        # Create a DataLoader for each label group.
        dataloaders = {}
        for label, samples in label_to_samples.items():
            input_ids = torch.stack([s["input_ids"] for s in samples])
            attention_mask = torch.stack([s["attention_mask"] for s in samples])
            # Convert the label tensor to a scalar tensor.
            labels_tensor = torch.tensor([s["labels"].item() for s in samples], dtype=torch.long)
            dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels_tensor)
            dataloaders[label] = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloaders


class LayerDataset(Dataset):
    def __init__(self, data_dir='data', wanted_labels=None, layer_index=0, data_type='both', row_neurons_flag=True):
        """
        Args:
            data_dir (str): Path to directory where activation files are stored.
            wanted_labels (list): List of sentiment labels to include (e.g., ["anger", "fear"]).
            layer_index (int): layer index
            data_type (str): "activations", "gradients", or "both" (default: "both").
        """
        if wanted_labels is None:
            raise ValueError("You must provide a list of wanted labels.")

        if data_type not in ["activations", "gradients", "both"]:
            raise ValueError("data_type must be 'activations', 'gradients', or 'both'.")

        self.data_dir = data_dir
        self.data_type = data_type
        # self.layer_indexes = layer_indexes if layer_indexes else None

        # Get label mappings
        self.label_mapping, self.labels = get_wanted_label(wanted_labels)

        self.all_data = []  # Store (activation, gradient, label, layer_idx)

        # Load each sentiment file
        for sentiment in self.labels:
            file_path = os.path.join(data_dir, f"activations_grads_{sentiment}_layer{layer_index}.pt")

            if not os.path.exists(file_path):
                print(f"⚠️ Warning: {file_path} not found. Skipping this sentiment.")
                continue

            try:
                data = torch.load(file_path, map_location="cpu")
                activations = data["activations"]  # (n_layers, n_samples, lang_input_d, lang_d)
                gradients = data["gradients"]  # Same shape
                L, N, I, d = activations.shape
                # if row_neurons_flag:
                #     N = N * I ## Activations are row activations - meaning each row is a new sample
                #     activations = activations.reshape(L, N, d)
                #     gradients = gradients.reshape(L, N, d)


                label_idx = self.label_mapping[self.labels.index(sentiment)]  # Convert sentiment to numeric label

                # Select only specified layers
                # selected_layers = self.layer_indexes if self.layer_indexes else list(range(L))

                # Flatten dataset: Iterate over selected layers and samples
                # for layer_idx in selected_layers:
                for sample_idx in range(N):
                    activation = activations[0, sample_idx] if self.data_type in ["activations",
                                                                                          "both"] else None
                    gradient = gradients[0, sample_idx] if self.data_type in ["gradients", "both"] else None
                    self.all_data.append((activation, gradient, label_idx, layer_index))

                print(f"✅ Loaded {sentiment}: {activations.shape}")

            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        activation, gradient, label, layer_idx = self.all_data[idx]

        if self.data_type == "activations":
            return activation, label, layer_idx
        elif self.data_type == "gradients":
            return gradient, label, layer_idx
        else:
            return activation, gradient, label, layer_idx


if __name__ == "__main__":
    wanted_labels = ["anger", "fear"]
