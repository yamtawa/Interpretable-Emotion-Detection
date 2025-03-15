import torch
import torch.nn as nn
import torch.nn.functional as F
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

class SAE(nn.Module):
    def __init__(self, input_dim, d_feat_scale=2):
        super(SAE, self).__init__()
        self.input_dim = input_dim
        self.d_feat_scale = d_feat_scale
        self.d_feat_dim = int(self.input_dim * self.d_feat_scale)

        self.Encoder = nn.Linear(self.input_dim, self.d_feat_dim, bias=True)

        self.Decoder = nn.Linear(self.d_feat_dim, self.input_dim, bias=False)
        # self.Decoder.weight = nn.Parameter(self.Encoder.weight.T)
        nn.init.orthogonal_(self.Decoder.weight)
    def forward(self, x):
        B, d = x.shape
        c = F.relu(self.Encoder(x)) ### (B, d_feat_dim)

        x_hat = self.Decoder(c) ### (B, input_dim)

        F_matrix = self.Decoder.weight.T ### (d_feat_dim, input_dim)

        return x_hat, c, F_matrix

class SAELoss(nn.Module):
    def __init__(self, alpha_sparsity=0.1):
        super(SAELoss, self).__init__()
        self.alpha_sparsity = alpha_sparsity
        self.mse_loss = nn.MSELoss()
        self.total_losses = []
        self.reconstruction_losses = []
        self.sparsity_losses = []

    def forward(self, x_hat, x, c):

        B, feat_dim = c.shape
        loss_recon = self.mse_loss(x_hat, x)  # L2 loss (MSE)
        loss_sparsity = torch.norm(c, p=1) / (feat_dim * B) # L1 sparsity loss
        loss = loss_recon + self.alpha_sparsity * loss_sparsity
        self.total_losses.append(loss.item())
        self.reconstruction_losses.append(loss_recon.item())
        # self.sparsity_losses.append(loss_sparsity.item())
        return loss, loss_recon, loss_sparsity