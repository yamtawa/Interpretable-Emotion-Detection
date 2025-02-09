import torch.nn as nn
from transformers import  BertModel


class EmotionBERT(nn.Module):
    def __init__(self, model_name = "bert-base-uncased", num_labels=27,fine_tune_only=True):
        super(EmotionBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.fine_tune_only=fine_tune_only
        if self.fine_tune_only:
            for param in self.bert.parameters():
                param.requires_grad = False
            self.bert.eval()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # Classification on CLS token
        return logits, outputs.hidden_states  # Returning all hidden states for analysis
