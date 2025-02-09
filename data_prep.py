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


def get_dataloader(dataset_name='go_emotions', batch_size=16, split='train'):
    dataset = EmotionDataset(dataset_name=dataset_name, split=split)

    if split == "test":
        test_size = len(dataset) // 2
        eval_size = len(dataset) - test_size
        eval_dataset, test_dataset = random_split(dataset, [eval_size, test_size])
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return eval_loader, test_loader

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
