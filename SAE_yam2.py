import os
from collections import defaultdict

import numpy as np
import torch.optim as optim
from cluster_P0 import cluster_and_plot
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torchmetrics.classification import Accuracy
from explore_nurons import load_pkl2dict
from load_config import load_config
class SparseAutoencoder_Linear(nn.Module):
    def __init__(self, input_size=768, latent_channels=800,lambda_sparse=1):
        super( SparseAutoencoder_Linear, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, latent_channels * input_size),
            nn.ELU(),
        )

        self.decoder = nn.Linear(latent_channels * input_size, input_size)

        self.latent_channels = latent_channels
        self.input_size = input_size
        self.lambda_sparse=lambda_sparse

    def forward(self, x):
        batch_size = x.size(0)
        # Encoding

        batch_size = x.size(0)

        x_flat = x.view(batch_size, self.input_size[0], -1)  # [batch_size, input_channels, 128 * 768]
        P = torch.einsum('bic,cih->bih', x_flat,
                         self.pixel_weights_encoder) + self.bias_encoder  # [batch_size, 128*768, latent_channels]
        P = P.permute(0, 2, 1).view(batch_size, self.latent_channels, self.input_size[1],
                                    self.input_size[2])  # [batch_size, latent_channels, 128, 768]

        # P = self.encoder(x)  # [b, 800 * input_size]
        # P = P.view(batch_size, self.latent_channels, self.input_size)  # [b, 800, input_size]

        # Decoding
        P_flat = P.view(batch_size, -1)
        output = self.decoder(P_flat)  # [b, input_size]

        return output, P

    def sparse_reconstruction_loss(self,input, output, P):
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(output, input)

        # Sparsity loss (encourage sparsity along rows of P)
        sparsity_loss = torch.mean(torch.abs(P))

        total_loss = reconstruction_loss + self.lambda_sparse * sparsity_loss
        return total_loss,sparsity_loss


class SparseAutoencoder_Conv(nn.Module):
    def __init__(self, input_size=(1, 128, 768), latent_channels=300, lambda_sparse=1,num_classes=5):
        super(SparseAutoencoder_Conv, self).__init__()
        height, width = input_size[1], input_size[2]


        self.pixel_weights_encoder = nn.Parameter(
            torch.randn(height * width, latent_channels)  # Per-pixel independent weights
        )
        self.bias_encoder = nn.Parameter(torch.zeros(height * width, latent_channels))

        self.pixel_weights = nn.Parameter(
            torch.randn(latent_channels, input_size[1] * input_size[2])
        )
        self.bias = nn.Parameter(torch.zeros(input_size[1] * input_size[2]))

        self.latent_channels = latent_channels
        self.input_size = input_size
        self.lambda_sparse = lambda_sparse
        self.mlp_head=nn.Linear(height * width, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        x_flat = x.view(batch_size, -1)  # [batch_size, 128 * 768]
        P = torch.einsum('bi,il->bil', x_flat,
                         self.pixel_weights_encoder) + self.bias_encoder  # [batch_size, 128 * 768, latent_channels]

        P = P.permute(0, 2, 1).view(batch_size, self.latent_channels, self.input_size[1],
                                    self.input_size[2])  # [batch_size, latent_channels, 128, 768]
        P=self.sharp_sigmoid(P)
        logits=self.mlp_head((P[:,0]*x.squeeze(1)).reshape(batch_size,P.shape[-1]*P.shape[-2]))

        P_flat = P.view(batch_size, self.latent_channels, -1)  # [batch_size, latent_channels, 128 * 768]
        output = torch.einsum('bci,ci->bi', P_flat, self.pixel_weights) + self.bias # [batch_size, 128 * 768, 128 * 768]
        output = output.view(batch_size, self.input_size[0], self.input_size[1], self.input_size[2])  # [batch_size, 1, 128, 768]

        return output, P, logits

    def sharp_sigmoid(self,x, threshold=0.8, sharpness=20):
        return torch.sigmoid(sharpness * (x - threshold))

    def sparse_reconstruction_loss(self, input, output, P, logits, targets,P_mean_d=[],start_p0_loss=False):
        reconstruction_loss = F.mse_loss(output, input)
        rho = 0.0000001  # desired sparse activation level
        eps = 1e-10
        # batch_p0_inner_loss=torch.zeros(1).to(device)
        batch_p0_loss=torch.zeros(1).to(device)
        if start_p0_loss:
            pp_cls=P[:, 0]
            batch_p0_loss+=(torch.abs(pp_cls - P_mean_d[-1])).mean()

        rho_hat = torch.mean(P, dim=0)
        sparsity_loss = torch.mean(rho * torch.log((rho + eps) / (rho_hat + eps)) +
                                     (1 - rho) * torch.log((1 - rho + eps) / (1 - rho_hat + eps)))
        P_centered = P - P.mean(dim=(2, 3), keepdim=True)  # Center around channel mean

        # Compute the pairwise cosine similarity between channels (flatten spatial dims)
        P_flat = P_centered.view(P.size(0), P.size(1), -1)  # [batch_size, channels, spatial_dim]
        similarity = torch.matmul(P_flat, P_flat.transpose(1, 2))  # [batch_size, channels, channels]

        # Mask out diagonal (self-similarity) and compute mean similarity
        mask = torch.eye(similarity.size(1), device=similarity.device).unsqueeze(0)
        diversity_loss = torch.mean(similarity * (1 - mask))
        loss_fn = nn.CrossEntropyLoss()
        cls_loss = loss_fn(logits, targets)
        # p0_loss= 0.1*batch_p0_inner_loss+ batch_p0_loss

        total_loss = 10*reconstruction_loss + sparsity_loss +0.000005*cls_loss + batch_p0_loss #0.001*diversity_loss

        return total_loss, sparsity_loss


def training_loop(model, dataloader,targets_names, optimizer, scheduler, num_class, epochs=10, device='cpu', run_name='default'):
    model.to(device)
    model.train()

    loss_history = []
    loss_history_sparse = []
    accuracy_history = []  # Store accuracy over epochs
    P0_mean_l=[]
    accuracy = Accuracy(task="multiclass", num_classes=num_class).to(device)  # Move accuracy to device

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        total_loss_sparse = 0
        save_plot = True
        P0_batch_l = []

        # Reset accuracy at the beginning of each epoch
        accuracy.reset()

        for input_tensor, targets, indices in dataloader:
            input_tensor, targets= input_tensor.to(device), targets.to(device)
            optimizer.zero_grad()

            output, P, logits = model(input_tensor)
            for cls in targets.detach().unique():
                P0_batch_l.append(P[:, 0].mean(axis=0))
            if (epoch % 10 == 0 or epoch == epochs - 1) and save_plot:
                plot_reconstruction(input_tensor, output,targets_names[indices[0]], P,P0_mean_l[-1] if len(P0_mean_l)>0 else [] , epoch + 1)
                print(f"\nInput abs mean: {torch.abs(input_tensor).mean().item():.6f}")
                print(f"Output abs mean: {torch.abs(output).mean().item():.6f}")
                print(f"P abs mean: {P.mean().item():.6f}")
                print(f"Accuracy: {accuracy_history[-1] if len(accuracy_history)>0 else 0} ")
                save_plot = False

            # Compute loss and backward
            loss, sparsity_loss = model.sparse_reconstruction_loss(input_tensor, output, P, logits, targets,P0_mean_l,start_p0_loss=max(accuracy_history)>0.99 if len(accuracy_history)>0 else False)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_loss_sparse += sparsity_loss.item()

            # Update accuracy
            accuracy.update(logits, targets)

            if device == 'cuda':
                # Free up memory after each batch
                del input_tensor, P
                torch.cuda.empty_cache()


        last_4_elemnts=P0_mean_l[-4:]
        last_4_elemnts.append(torch.stack(P0_batch_l, dim=0).mean(axis=0))
        stacked = torch.stack(last_4_elemnts, dim=0)  # shape: (N, ...)
        weights = torch.softmax(torch.tensor([0.05, 0.05, 0.1, 0.15, 0.75][-len(last_4_elemnts):]), dim=0).to(device)
        weighted = stacked * weights.view(-1, *[1] * (stacked.ndim - 1))  # broadcast weights
        P0_mean_l.append(weighted.sum(dim=0).detach())

        # Compute average loss and accuracy for the epoch
        avg_loss = total_loss / len(dataloader)
        avg_loss_sparse = total_loss_sparse / len(dataloader)
        epoch_accuracy = accuracy.compute().item()  # Compute mean accuracy

        # Store results in history
        loss_history.append(avg_loss)
        loss_history_sparse.append(avg_loss_sparse)
        accuracy_history.append(epoch_accuracy)


        # Save model weights every 10 epochs or at the last epoch
        if epoch % 3== 0 or epoch == epochs - 1:
          torch.save(model.state_dict(), f'SAE_models_weights/{run_name}.pth')
    clusteres={}
    for cls in dataloader.dataset.tensors[1].unique():
        clusteres[f"label:{cls}"]=dataloader.dataset.tensors[0][dataloader.dataset.tensors[1] == cls].squeeze().detach().cpu() * (
                    P0_mean_l[-1] > 0.7).cpu()
    return loss_history, loss_history_sparse, accuracy_history,clusteres,P0_mean_l

def eval(model, dataloader,targets_names, num_class, epochs=10, device='cpu', run_name='default'):
    model.to(device)
    model.val()

    loss_history = []
    loss_history_sparse = []
    accuracy_history = []  # Store accuracy over epochs

    accuracy = Accuracy(task="multiclass", num_classes=num_class).to(device)  # Move accuracy to device

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        total_loss_sparse = 0
        save_plot = True

        # Reset accuracy at the beginning of each epoch
        accuracy.reset()

        for input_tensor, targets, indices in dataloader:
            input_tensor, targets= input_tensor.to(device), targets.to(device)

            output, P, logits = model(input_tensor)

            if (epoch % 10 == 0 or epoch == epochs - 1) and save_plot:
                plot_reconstruction(input_tensor, output,targets_names[indices[0]], P, epoch + 1)
                print(f"\nInput abs mean: {torch.abs(input_tensor).mean().item():.6f}")
                print(f"Output abs mean: {torch.abs(output).mean().item():.6f}")
                print(f"P abs mean: {P.mean().item():.6f}")
                print(f"Accuracy: {accuracy_history[-1] if len(accuracy_history)>0 else 0} ")
                save_plot = False

            accuracy.update(logits, targets)

            if device == 'cuda':
                # Free up memory after each batch
                del input_tensor, P
                torch.cuda.empty_cache()

        epoch_accuracy = accuracy.compute().item()  # Compute mean accuracy
        accuracy_history.append(epoch_accuracy)


    return accuracy_history
def create_dataloaders(data,one_key_only=False):

    if one_key_only:
        dataset = TensorDataset(
            data[one_key_only][5].unsqueeze(1))#[:100, :, 50:100, 100:150]  # sape of [num_samples_per_label,1,128,768]
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    data_tensors = []
    targets = []

    for key in data.keys():
        tensor = data[key][5].unsqueeze(1)#[:100, :, 50:100, 100:150]
        data_tensors.append(tensor)
        targets.extend([key] * tensor.size(0))  # Repeat the key as target for each sample
    data_tensor = torch.cat(data_tensors, dim=0)
    targets_tensor = torch.tensor([list(data.keys()).index(t) for t in targets], dtype=torch.long)
    indices = torch.randperm(data_tensor.size(0))
    data_tensor = data_tensor[indices]
    targets_names = [targets[t]for t in indices.tolist()]
    targets_tensor = targets_tensor[indices]
    dataset = TensorDataset(data_tensor, targets_tensor,indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader,targets_names
def split_data_dict(data):
    train_d = {}
    test_d = {}
    for key, tensor_data in data.items():
        split_idx = tensor_data.shape[1] // 2
        train_d[key] = tensor_data[:, :split_idx]
        test_d[key] = tensor_data[:, split_idx:]
    return train_d, test_d
def plot_reconstruction(input_tensor, output,targets_name, P,P0_mean, epoch):
    input_img = input_tensor.detach().cpu().numpy()[0, 0, 50: 100, 100: 150]
    output_img = output.detach().cpu().numpy()[0, 0, 50: 100, 100: 150]
    latent_img = P.detach().cpu().numpy()[0, 0, 50: 100, 100: 150]
    P0_mean_patch= P0_mean[50: 100, 100: 150].cpu() if P0_mean != [] else np.zeros_like(latent_img)
    os.makedirs(f'plot/{run_name}', exist_ok=True)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.title('Input')
    plt.imshow(input_img, aspect='auto', cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.imshow(output_img, aspect='auto', cmap='viridis')
    plt.title('Reconstruction')

    plt.subplot(1, 4, 3)
    plt.imshow(latent_img, aspect='auto', cmap='viridis')
    plt.title(f'Latent ,chanel 0 (emotion), Emotion-{targets_name}')

    plt.subplot(1, 4, 4)
    plt.imshow(P0_mean_patch, aspect='auto', cmap='viridis')
    plt.title(f'P0_mean ,Emotion-{targets_name}')

    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()

    plt.savefig(f'plot/{run_name}/sparse_feature_map_epoch_{epoch}.png')
    plt.close()

def plot_loss(loss_history,loss_history_sparse):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2,1)
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid()

    plt.subplot(1, 2,2)
    plt.plot(loss_history_sparse)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss-Sparse Over Epochs')
    plt.grid()

    plt.savefig(f'plot/{run_name}_loss_curves.png')
def plot_accuracy(accuracy_history):
    plt.figure(figsize=(12, 4))
    plt.plot(accuracy_history)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.grid()
    plt.savefig(f'plot/{run_name}_accuracy_curves.png')

def run():
    os.makedirs(rf'plots\{run_name}',exist_ok=True)
    os.makedirs(rf'SAE_models_weights',exist_ok=True)

    data=load_pkl2dict(path_to_pkl)
    train_d, test_d=split_data_dict(data)
    dataloader,targets_names=create_dataloaders(train_d)
   #TODO- make this controlable from CFG
    if wanted_model=="CONV":
        model = SparseAutoencoder_Conv(input_size=input_size, latent_channels=100,num_classes=num_class)
    elif wanted_model=="LINEAR":
        model = SparseAutoencoder_Linear(input_size=input_size[-1], latent_channels=100)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.000001)
    loss_history,sparsity_loss, accuracy_history,clusteres,P0_mean_d=training_loop(model, dataloader,targets_names, optimizer,scheduler, num_class, epochs=epochs,device=device)
    plot_loss(loss_history, sparsity_loss)
    plot_accuracy(accuracy_history)
    cluster_and_plot(clusteres)

# Example usage
if __name__ == '__main__':
    batch_size = 16
    input_size = (1,128,768)
    wanted_model="CONV"
    run_name='CONV_with_one_masked_label_1_4'
    epochs=200
    num_class=6

    CFG = load_config(config_name='config')
    path_to_pkl=r'saved_activations/grad_activations_dict.pkl'
    device='cuda' if torch.cuda.is_available() else 'cpu'
    run()




