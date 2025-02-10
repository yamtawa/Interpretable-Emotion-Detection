from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from transformers import BertTokenizer

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


