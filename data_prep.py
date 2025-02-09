import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import BertTokenizer


class GoEmotionsDataset(Dataset):
    def __init__(self, split='train', max_length=128):
        self.dataset = load_dataset("go_emotions", split=split)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        labels = self.dataset[idx]["labels"]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length,
                                return_tensors='pt')
        return {"input_ids": inputs["input_ids"].squeeze(), "attention_mask": inputs["attention_mask"].squeeze(),
                "labels": torch.tensor(labels, dtype=torch.float)}


def get_dataloader(batch_size=16, split='train'):
    dataset = GoEmotionsDataset(split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)