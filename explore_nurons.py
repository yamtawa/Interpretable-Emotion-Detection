import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import scipy.stats
from utils import get_wanted_label, get_dict_labels
from collections import defaultdict
from tqdm import tqdm
from visualizations import *
from models import *
import os


def get_function_from_name(function_name):
    if function_name == 'get_most_activated_neurons_per_label':
        return get_most_activated_neurons_per_label
    elif function_name == 'compute_neuron_correlation':
        return compute_neuron_correlation
    elif function_name == 'train_probing_classifier':
        return train_probing_classifier
    elif function_name == 'cluster_neurons':
        return cluster_neurons
    raise ValueError(
        "Unknown exploration function name. Choose one of [extract_neuron_activations,compute_neuron_correlation,train_probing_classifier,compute_saliency,cluster_neurons]")

# TODO - Consider the case when you want to differentiate between true predictions and false predictions of a label.

def loop_batches(model, dataloaders, device, criterion, wanted_labels='all', save_flag=True,
                 save_dir='data', layer_index=0):

    keys,wanted_labels = get_wanted_label(wanted_labels)  # keys is a list of keys
    values = [dataloaders[key] for key in keys if key in dataloaders]

    neurons_all_activations = {}
    neurons_all_gradients = {}
    for idx, dataloader in enumerate(values):
        neuron_label_activation = defaultdict(lambda: None)
        neuron_label_gradients = defaultdict(lambda: None)

        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                                     desc=f"Collecting activations and gradients from language model with label {wanted_labels[idx]}...", ascii=True):

            data = batch[0].to(device)
            attention_mask = batch[1].to(device)
            target = batch[2].to(device)
            model.zero_grad()
            logits, hidden_states = model(data, attention_mask)
            for layer in hidden_states:
                layer.retain_grad()
            loss = criterion(logits, target)
            loss.backward()
            for layer_idx, layer in tqdm(enumerate(hidden_states)):
                if layer_idx<layer_index:
                    continue
                elif layer_idx>layer_index:
                    break
                layer_cpu = layer.clone().detach().cpu()  # Move to CPU for good memory allocation
                grad_cpu = layer.grad.clone().detach().cpu()
                if neuron_label_activation[layer_idx] is None:
                    neuron_label_activation[layer_idx] = layer_cpu
                    neuron_label_gradients[layer_idx] = grad_cpu
                else:
                    neuron_label_activation[layer_idx] = torch.cat((neuron_label_activation[layer_idx], layer_cpu),dim=0)
                    neuron_label_gradients[layer_idx] = torch.cat((neuron_label_gradients[layer_idx], grad_cpu), dim=0)
            if device=='cuda':
                # Free up memory after each batch
                del data, attention_mask, target, logits, hidden_states
                torch.cuda.empty_cache()  # Clear CUDA cache
        neurons_all_activations[wanted_labels[idx]] = torch.stack(list(neuron_label_activation.values())) ### dict of sentiments. For each sentiment activation shape: (n_layers, n_samples, lang_input_d, lang_hidden))
        neurons_all_gradients[wanted_labels[idx]] = torch.stack(list(neuron_label_gradients.values()))
        if save_flag:
            os.makedirs(save_dir, exist_ok=True)
            filename = 'activations_grads_' + wanted_labels[idx] + f'_layer{layer_index}.pt'
            save_dict = {
                "activations": neurons_all_activations[wanted_labels[idx]][0].unsqueeze(0),  # Dictionary of {label: tensor}, saving 0 layer only
                "gradients": neurons_all_gradients[wanted_labels[idx]][0].unsqueeze(0)  # Dictionary of {label: tensor}, saving 0 layer only
            }
            torch.save(save_dict, os.path.join(save_dir, filename))
            print(f'Saved {len(dataloader.dataset)} activations and grads for {wanted_labels[idx]} to file: {os.path.join(save_dir, filename)}')



def predict_layer_activation(model, dataloader, device, wanted_labels=['anger','fear'], N=128, layer_idx=0, scale_str='2'):
    model.eval()
    keys, wanted_labels = get_wanted_label(wanted_labels)
    c_fear_sum = 0
    fear_counter = 0
    c_anger_sum = 0
    anger_counter = 0
    for idx, layer in tqdm(enumerate(dataloader), desc="Testing progress", ascii=True):
        activation, label = layer
        sentiment = wanted_labels[label[0].item()]
        activation = (activation - activation.mean(dim=1, keepdim=True)) / (activation.std(dim=1, keepdim=True) + 1e-5)
        activation = activation.clamp(-5, 5)  # Clip extreme values to prevent instability
        activation = activation.to(device)
        x_hat, c, F_matrix = model(activation)
        x_hat, c, F_matrix = x_hat.detach().cpu().numpy(), c.detach().cpu().numpy(), F_matrix.detach().cpu().numpy()
        # visualize_c_heatmap(c, sentiment, save_path=f'figures/testing_fig_{scale_str}_{sentiment}.png', scale_str=scale_str)
        if label[0].item() == 0:
            c_anger_sum += c
            anger_counter += 1
        else:
            c_fear_sum += c
            fear_counter += 1
    c_anger_avg = c_anger_sum / anger_counter
    c_fear_avg = c_fear_sum / fear_counter
    visualize_c_heatmap(c_anger_avg, 'anger', save_path=f'figures/anger_avg_layer{layer_idx}', scale_str=scale_str)
    visualize_c_heatmap(c_fear_avg, 'fear', save_path=f'figures/fear_avg_layer{layer_idx}', scale_str=scale_str)

    a=5


def get_most_activated_neurons_per_label(neurons_all_activations: dict,d_labels:dict):
    d = {}
    for label, activations in neurons_all_activations.items():
        d[label] = activations.mean(axis=1)
    return d




def compute_neuron_correlation(neuron_activations, emotion_labels):  # TODO - This is the scratch it doesnt work yet

    correlations = [scipy.stats.pearsonr(neuron_activations[:, i], emotion_labels)[0] for i in
                    range(neuron_activations.shape[1])]

    return correlations


def train_probing_classifier(activations, labels):  # TODO - This is the scratch it doesnt work yet

    clf = LogisticRegression(max_iter=1000)
    clf.fit(activations, labels)

    return clf.coef_  # Feature importance per neuron


def cluster_neurons(neuron_activations, labels, num_clusters=6):  # TODO - This is the scratch it doesnt work yet

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(neuron_activations)

    return cluster_labels  # Each neuron gets assigned to a cluster

if __name__ == "__main__":
    file_paths = ["data/activations_grads_anger.pt", "data/activations_grads_fear.pt"]
