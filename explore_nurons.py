import os

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import scipy.stats
from utils import get_wanted_label,get_dict_labels
from collections import defaultdict
import pickle


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

def load_pkl2dict(dict_path):
    with open(dict_path, 'rb') as f:
        dic = pickle.load(f)
    return dic

def save_dict2pkl(dict_path,dict):
    os.makedirs(os.path.dirname(dict_path),exist_ok=True)
    with open(dict_path, 'wb') as f:
        pickle.dump(dict, f, protocol=pickle.HIGHEST_PROTOCOL)

# TODO - Consider the case when you want to differentiate between true predictions and false predictions of a label.

def loop_batches(model, dataloaders, device, criterion, function_name='extract_neuron_activations',wanted_labels='all',saved_activations_path=r'saved_activations'):
    exp_func = get_function_from_name(function_name)
    keys,wanted_labels = get_wanted_label(wanted_labels)  # keys is a list of keys
    values = [dataloaders[key] for key in keys if key in dataloaders]
    neurons_all_activations = {}
    neurons_all_gradients = {}
    activations_dict_path=os.path.join(saved_activations_path,'activations_dict.pkl')
    grad_activations_dict_path=os.path.join(saved_activations_path,'grad_activations_dict.pkl')

    if os.path.isfile(activations_dict_path) and os.path.isfile(grad_activations_dict_path):
        neurons_all_activations=load_pkl2dict(activations_dict_path)
        neurons_all_gradients = load_pkl2dict(grad_activations_dict_path)
    else:
        for idx, dataloader in enumerate(values):
            neuron_label_activation = defaultdict(lambda: None)
            neuron_label_gradients = defaultdict(lambda: None)
            for batch in dataloader:
                data = batch[0].to(device)
                attention_mask = batch[1].to(device)
                target = batch[2].to(device)
                model.zero_grad()
                logits, hidden_states = model(data, attention_mask)
                for layer in hidden_states:
                    layer.retain_grad()
                loss = criterion(logits, target)
                loss.backward()
                for layer_idx, layer in enumerate(hidden_states):
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
            neurons_all_activations[wanted_labels[idx]] = torch.stack(list(neuron_label_activation.values()))
            neurons_all_gradients[wanted_labels[idx]] = torch.stack(list(neuron_label_gradients.values()))
        save_dict2pkl(activations_dict_path, neurons_all_activations)
        save_dict2pkl(grad_activations_dict_path, neurons_all_gradients)

    d_labels=get_dict_labels(neurons_all_activations,keys,wanted_labels)
    exp_func(neurons_all_activations,d_labels)


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
