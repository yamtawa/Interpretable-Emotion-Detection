import os
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    def __init__(self, input_size=(1, 128, 768), latent_channels=300, lambda_sparse=1):
        super(SparseAutoencoder_Conv, self).__init__()
        height, width = input_size[1], input_size[2]

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=input_size[0], out_channels=latent_channels, kernel_size=1),
        #     nn.Sigmoid(),
        # )

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

    def forward(self, x):
        batch_size = x.size(0)

        x_flat = x.view(batch_size, -1)  # [batch_size, 128 * 768]
        P = torch.einsum('bi,il->bil', x_flat,
                         self.pixel_weights_encoder) + self.bias_encoder  # [batch_size, 128 * 768, latent_channels]

        P = P.permute(0, 2, 1).view(batch_size, self.latent_channels, self.input_size[1],
                                    self.input_size[2])  # [batch_size, latent_channels, 128, 768]
        P=torch.sigmoid(P)
        # Encoding
        # P = self.encoder(x)  # [batch_size, latent_channels, 128, 768]

        P_flat = P.view(batch_size, self.latent_channels, -1)  # [batch_size, latent_channels, 128 * 768]
        output = torch.einsum('bci,ci->bi', P_flat, self.pixel_weights) + self.bias # [batch_size, 128 * 768, 128 * 768]
        output = output.view(batch_size, self.input_size[0], self.input_size[1], self.input_size[2])  # [batch_size, 1, 128, 768]

        return output, P



    def sparse_reconstruction_loss(self, input, output, P):
        reconstruction_loss = F.mse_loss(output, input)
        rho = 0.00001  # desired sparse activation level
        eps = 1e-10
        # P_sigmoid = torch.sigmoid(P)
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
        total_loss = reconstruction_loss + 10*sparsity_loss + 0.1*diversity_loss

        return total_loss, sparsity_loss


def training_loop(model, dataloader, optimizer,scheduler, epochs=10, device='cpu'):
    model.to(device)
    model.train()
    loss_history = []
    loss_history_sparse = []


    for epoch in tqdm(range(epochs)):
        total_loss = 0
        total_loss_sparse=0
        save_plot=True

        for batch in dataloader:
            input_tensor = batch[0].to(device)
            optimizer.zero_grad()
            output, P = model(input_tensor)
            if (epoch % 100 == 0 or epoch == epochs - 1) and save_plot:
                plot_reconstruction(input_tensor, output, P, epoch + 1)
                print(f"\nInput abs mean: {torch.abs(input_tensor).mean().item():.6f}")
                print(f"Output abs mean: {torch.abs(output).mean().item():.6f}")
                print(f"P abs mean: {P.mean().item():.6f}")
                save_plot = False
            loss,sparsity_loss = model.sparse_reconstruction_loss(input_tensor, output, P)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_loss_sparse+=sparsity_loss.item()

            if device == 'cuda':
                # Free up memory after each batch
                del input_tensor, P
                torch.cuda.empty_cache()

        avg_loss = total_loss / len(dataloader)
        avg_loss_sparse = total_loss_sparse / len(dataloader)

        loss_history.append(avg_loss)
        loss_history_sparse.append(avg_loss_sparse)

        if epoch  % 10 == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), f'SAE_models_weights\{run_name}.pth')
    return loss_history, loss_history_sparse


def plot_reconstruction(input_tensor, output, P, epoch):
    input_img = input_tensor.detach().cpu().numpy()[0, 0]
    output_img = output.detach().cpu().numpy()[0, 0]
    latent_img = P.detach().cpu().numpy()[0, torch.randint(0, P.shape[1], (1,)).item()]

    os.makedirs(f'plot/{run_name}', exist_ok=True)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title('Input')
    plt.imshow(input_img, aspect='auto', cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(output_img, aspect='auto', cmap='viridis')
    plt.title('Reconstruction')

    plt.subplot(1, 3, 3)
    plt.imshow(latent_img, aspect='auto', cmap='viridis')
    plt.title('Latent (Random Channel)')

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

def run():
    os.makedirs(rf'plots\{run_name}',exist_ok=True)
    os.makedirs(rf'SAE_models_weights',exist_ok=True)

    data=load_pkl2dict(path_to_pkl)
    dataset = TensorDataset(data['anger'][5].unsqueeze(1)[:100,:,50:100,100:150])  # sape of [num_samples_per_label,1,128,768]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
   #TODO- make this controlable from CFG
    if wanted_model=="CONV":
        model = SparseAutoencoder_Conv(input_size=input_size, latent_channels=100)
    elif wanted_model=="LINEAR":
        model = SparseAutoencoder_Linear(input_size=input_size[-1], latent_channels=100)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.000001)
    loss_history,sparsity_loss=training_loop(model, dataloader, optimizer,scheduler, epochs=epochs,device=device)
    plot_loss(loss_history, sparsity_loss)

# Example usage
if __name__ == '__main__':
    batch_size = 16
    input_size = (1,50,50)
    wanted_model="CONV"
    run_name='try1_CONV_no_labels_one_chanel'
    epochs=5000

    CFG = load_config(config_name='config')
    path_to_pkl=r'saved_activations/activations_dict.pkl'
    device='cuda' if torch.cuda.is_available() else 'cpu'
    run()




