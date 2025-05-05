import json
import os
import numpy as np
from cluster_P0 import cluster_and_plot
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torchmetrics.classification import Accuracy
from explore_nurons import load_pkl2dict
from kmens_page import preform_kmeans
from load_config import load_config



class SparseInputMask(nn.Module):
    def __init__(self, height, width,n_ch, k=1, tau=0.1):
        """
        Learns a binary mask of shape [1, height, width] with exactly `k` active pixels.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.n_channels=n_ch
        self.k = k
        self.tau = tau
        self.logits = nn.Parameter(torch.randn(self.n_channels, self.height, self.width))  # learnable logits

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.height and W == self.width and C== self.n_channels, "Input size mismatch with mask"

        # Flatten for top-k sampling
        flat_logits = self.logits.view(-1)  # shape: [C* H * W]
        gumbel_noise = -torch.empty_like(flat_logits).exponential_().log()
        scores = (flat_logits + gumbel_noise) / self.tau

        # Get top-k mask
        topk_vals, topk_indices = torch.topk(scores, self.k)
        hard_mask = torch.zeros_like(flat_logits)
        hard_mask[topk_indices] = 1.0

        # Straight-through estimator
        soft_mask = F.softmax(flat_logits / self.tau, dim=0)
        final_mask = (hard_mask - soft_mask).detach() + soft_mask

        # Reshape back to [1, H, W] and broadcast over batch and channel
        binary_mask = final_mask.view(C, H, W)  # shape: [1, H, W]
        binary_mask = binary_mask.unsqueeze(0).expand(B, C, H, W)  # shape: [B, C, H, W]

        return x * binary_mask, binary_mask  # pixel-wise masked input


class SparseAutoencoder_Conv(nn.Module):
    def __init__(self, input_size=(1, 128, 768), latent_channels=1,num_classes=5):
        super(SparseAutoencoder_Conv, self).__init__()
        ch, height, width = input_size[0], input_size[1], input_size[2]

        self.height = height
        self.width = width
        self.n_channels = ch

        self.latent_channels = latent_channels
        self.input_size=input_size

        self.input_learnable_mask = SparseInputMask(self.height , self.width,self.n_channels, k=ACTIVE_NEURONS_COUNT)
        self.mlp_head=nn.Linear(height * width*self.n_channels, num_classes)
        self.k=ACTIVE_NEURONS_COUNT


    def forward(self, x):
        batch_size = x.size(0)

        mask_output, binary_mask=self.input_learnable_mask(x)

        mask_output_flat=mask_output.reshape(batch_size, self.height * self.width * self.n_channels)

        logits=self.mlp_head(mask_output_flat)

        return  binary_mask, logits



    def sparse_reconstruction_loss(self, P, logits, targets,P_mean_d,start_po_loss):

        batch_p0_loss=torch.zeros(1).to(device)
        if start_po_loss:
            pp_cls=P[:, 0]
            batch_p0_loss+=(torch.abs(pp_cls - (P_mean_d[-1]   if P_mean_d !=[] else P.mean(axis=0)))).mean()
        loss_fn = nn.CrossEntropyLoss()
        cls_loss = loss_fn(logits, targets)
        total_loss = cls_loss + 0.1*batch_p0_loss

        return total_loss

def gumbel_topk(logits, k, tau=1.0, hard=True):
    # Add Gumbel noise
    noise = -torch.empty_like(logits).exponential_().log()
    gumbel_logits = (logits + noise) / tau

    # Get top-k indices
    topk_vals, topk_indices = torch.topk(gumbel_logits, k, dim=-1)

    # Create binary mask (1 at top-k locations)
    hard_mask = torch.zeros_like(logits)
    hard_mask.scatter_(-1, topk_indices, 1.0)

    if hard:
        return (hard_mask - gumbel_logits.softmax(dim=-1)).detach() + gumbel_logits.softmax(dim=-1)
    else:
        return gumbel_logits.softmax(dim=-1)  # no top-k hard cutoff


def training_loop(model, dataloader,targets_names, optimizer, scheduler, num_class, epochs=10, device='cpu', run_name='default'):
    model.to(device)
    model.train()

    loss_history = []
    accuracy_history = []  # Store accuracy over epochs
    P0_mean_l=[]
    accuracy = Accuracy(task="multiclass", num_classes=num_class).to(device)  # Move accuracy to device

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        save_plot = True
        P0_batch_l = []

        # Reset accuracy at the beginning of each epoch
        accuracy.reset()

        for input_tensor, targets, indices in dataloader:
            input_tensor, targets= input_tensor.to(device), targets.to(device)
            optimizer.zero_grad()

            P, logits = model(input_tensor)
            P0_batch_l.append(P.mean(axis=0))
            if (epoch % 2 == 0 or epoch == epochs - 1) and save_plot:
                rows, col = np.where(P0_batch_l[-1][0].detach().cpu().numpy()!=0)
                result = np.column_stack((rows, col))
                print(f"Chosen neurons locations: \n{result}")
                print(f"Accuracy: {accuracy_history[-1] if len(accuracy_history)>0 else 0} ")
                save_plot = False

            # Compute loss and backward
            loss= model.sparse_reconstruction_loss( P, logits, targets,P0_mean_l,accuracy_history[-1] >0.33 if len(accuracy_history)>0 else False )
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

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
        epoch_accuracy = accuracy.compute().item()  # Compute mean accuracy

        # Store results in history
        loss_history.append(avg_loss)
        accuracy_history.append(epoch_accuracy)


        # Save model weights every 10 epochs or at the last epoch
        if epoch % 3== 0 or epoch == epochs - 1:
          torch.save(model.state_dict(), f'SAE_models_weights/{run_name}.pth')
    clusteres={}
    _, indices = torch.topk(P0_mean_l[-1].flatten(), model.k)
    for cls in dataloader.dataset.tensors[1].unique():
        samples = dataloader.dataset.tensors[0][dataloader.dataset.tensors[1] == cls]  # shape: [B, 1, H, W]
        flat_samples = samples.view(samples.shape[0], -1)
        mask=P0_mean_l[-1] >0.8# [B, H*W]
        clusteres[f"label:{cls}"] = flat_samples[:, indices.detach().cpu()] if len(indices)>=mask.sum() else  flat_samples[:, mask.flatten().detach().cpu()]
    with open('wanted_indices_768.json', 'w') as f:
        json.dump(indices.tolist(), f)

    return loss_history, accuracy_history,clusteres,P0_mean_l


def create_dataloaders(data,one_key_only=False):

    if one_key_only:
        dataset = TensorDataset(
            data[one_key_only][5].unsqueeze(1))#[:100, :, 50:100, 100:150]  # sape of [num_samples_per_label,1,128,768]
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    data_tensors = []
    targets = []

    for key in data.keys():
        tensor = data[key][WANTED_LAYER].unsqueeze(1) if len(data[key][WANTED_LAYER].shape)==3 else data[key][WANTED_LAYER] #[:100, :, 50:100, 100:150]
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
    merged = { k: torch.stack([d[k] for d in data], dim=2)for k in data[0].keys()}
    for key, tensor_data in merged.items():
        split_idx = tensor_data.shape[1] // 2
        train_d[key] = tensor_data[:, :split_idx]
        test_d[key] = tensor_data[:, split_idx:]
    return train_d, test_d
# def plot_selected_neurons(targets_names, P0_batch,P0_mean , epoch,kernel=(100,100) ):
#     batch_=

def plot_loss(loss_history,):
    plt.figure(figsize=(12, 4))

    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid()

    plt.savefig(f'plots/{run_name}_loss_curves.png')
def plot_accuracy(accuracy_history):
    plt.figure(figsize=(12, 4))
    plt.plot(accuracy_history)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.grid()
    plt.savefig(f'plots/{run_name}_accuracy_curves.png')

def run():
    os.makedirs(rf'plots',exist_ok=True)
    os.makedirs(rf'SAE_models_weights',exist_ok=True)

    data_l=[load_pkl2dict(path) for path in path_to_pkl]
    train_d, test_d=split_data_dict(data_l)
    dataloader,targets_names=create_dataloaders(train_d)
   #TODO- make this controlable from CFG
    if wanted_model=="CONV":
        model = SparseAutoencoder_Conv(input_size=input_size, latent_channels=1,num_classes=num_class)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0000001)
    loss_history, accuracy_history,clusteres,P0_mean_d=training_loop(model, dataloader,targets_names, optimizer,scheduler, num_class, epochs=epochs,device=device)
    plot_loss(loss_history)
    plot_accuracy(accuracy_history)
    if PLOT_CLUSTERING:
        cluster_and_plot(clusteres,saved_name=run_name)
    preform_kmeans(clusteres, run_name, K=6)
def main(dummy1, dummy2):
    global batch_size,input_size,wanted_model,run_name,epoch,num_class,ACTIVE_NEURONS_COUNT,WANTED_LAYER,CFG,path_to_pkl,device, PLOT_CLUSTERING
    batch_size = 16
    input_sizes = [(1,128,768)]#,(1,128,768),(2,128,768)]
    wanted_model="CONV"
    run_names=['one_mask_for_all_grads_only']#'best_neurons_activation_only','one_mask_for_all_grads_only','one_mask_for_all_activation_and_gards']
    epochs=1000
    num_class=6
    ACTIVE_NEURONS_COUNTS=[1000]
    WANTED_LAYERS=[-1]
    CFG = load_config(config_name='config')
    path_to_pkls=[[r'saved_activations/grad_activations_dict.pkl']]#,[r'saved_activations/activations_dict.pkl',r'saved_activations/activations_dict.pkl']]
    device='cuda' if torch.cuda.is_available() else 'cpu'

    for path_to_pkl,input_size,run_name_partial in zip(path_to_pkls,input_sizes,run_names):
         for WANTED_LAYER in WANTED_LAYERS:
             for ACTIVE_NEURONS_COUNT in ACTIVE_NEURONS_COUNTS:
                 run_name=run_name_partial+f"_block{WANTED_LAYER}_{ACTIVE_NEURONS_COUNT}activated_neurons"
                 PLOT_CLUSTERING=True if ACTIVE_NEURONS_COUNT>1 else False
                 run()
# Example usage
if __name__ == '__main__':
    batch_size = 16
    input_sizes = [(1,128,768)]#,(1,128,768),(2,128,768)]
    wanted_model="CONV"
    run_names=['one_mask_for_all_grads_only']#'best_neurons_activation_only','one_mask_for_all_grads_only','one_mask_for_all_activation_and_gards']
    epochs=1000
    num_class=6
    ACTIVE_NEURONS_COUNTS=[1000]
    WANTED_LAYERS=[-1]
    CFG = load_config(config_name='config')
    path_to_pkls=[[r'saved_activations/grad_activations_dict.pkl']]#,[r'saved_activations/activations_dict.pkl',r'saved_activations/activations_dict.pkl']]
    device='cuda' if torch.cuda.is_available() else 'cpu'

    for path_to_pkl,input_size,run_name_partial in zip(path_to_pkls,input_sizes,run_names):
         for WANTED_LAYER in WANTED_LAYERS:
             for ACTIVE_NEURONS_COUNT in ACTIVE_NEURONS_COUNTS:
                 run_name=run_name_partial+f"_block{WANTED_LAYER}_{ACTIVE_NEURONS_COUNT}activated_neurons"
                 PLOT_CLUSTERING=True if ACTIVE_NEURONS_COUNT>1 else False
                 run()




