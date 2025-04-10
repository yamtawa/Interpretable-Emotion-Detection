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
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import defaultdict
from scipy.stats import ttest_ind, pearsonr, spearmanr
import matplotlib.pyplot as plt


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
        if idx > 1:
            continue
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

def extract_SAE_info(data_loader, type, model, device, wanted_labels):
    c_anger_all = []
    c_fear_all = []
    c_joy_all = []
    c_love_all = []
    c_sadness_all = []
    c_surprise_all = []


    for idx, layer in tqdm(enumerate(data_loader), desc=f"Extracting {type} info", ascii=True):
        activation, label = layer
        sentiment = wanted_labels[label[0].item()]
        activation = (activation - activation.mean(dim=1, keepdim=True)) / (activation.std(dim=1, keepdim=True) + 1e-5)
        activation = activation.clamp(-5, 5)  # Clip extreme values to prevent instability
        activation = activation.to(device)
        x_hat, c, F_matrix = model(activation)
        x_hat, c, F_matrix = x_hat.detach().cpu().numpy(), c.detach().cpu().numpy(), F_matrix.detach().cpu().numpy()
        # visualize_c_heatmap(c, sentiment, scale_str=scale_str)
        if label[0].item()==0:
            c_anger_all.append(c)
        elif label[0].item()==1:
            c_fear_all.append(c)
        elif label[0].item()==2:
            c_joy_all.append(c)
        elif label[0].item()==3:
            c_love_all.append(c)
        elif label[0].item()==4:
            c_sadness_all.append(c)
        else:
            c_surprise_all.append(c)
    c_anger_all = np.stack(c_anger_all, axis=0)  # Shape: (num_anger_samples, c_dim)
    c_fear_all = np.stack(c_fear_all, axis=0)  # Shape: (num_fear_samples, c_dim)
    c_joy_all = np.stack(c_joy_all, axis=0)  # Shape: (num_fear_samples, c_dim)
    c_love_all = np.stack(c_fear_all, axis=0)  # Shape: (num_fear_samples, c_dim)
    c_sadness_all = np.stack(c_fear_all, axis=0)  # Shape: (num_fear_samples, c_dim)
    c_surprise_all = np.stack(c_fear_all, axis=0)  # Shape: (num_fear_samples, c_dim)



    return c_anger_all, c_fear_all, c_joy_all, c_love_all, c_sadness_all, c_surprise_all

def predict_layer_activation(model, val_dataloader, test_dataloader, device, wanted_labels=[ 'anger', 'fear', 'joy', 'love', 'sadness', 'surprise'] , N=128, layer_idx=0, scale_str='2', alpha_str='025'):
    model.eval()
    keys, wanted_labels = get_wanted_label(wanted_labels)


    c_anger_val, c_fear_val, c_joy_val, c_love_val, c_sadness_val, c_surprise_val = extract_SAE_info(val_dataloader, type='val', model=model, device=device, wanted_labels=wanted_labels)
    c_anger_test, c_fear_test, c_joy_test, c_love_test, c_sadness_test, c_surprise_test = extract_SAE_info(test_dataloader, type='test', model=model, device=device, wanted_labels=wanted_labels)



    # visualize_F_heatmap(F_matrix[:20, :20], sentiment)
    c_anger_avg_val = c_anger_val.mean(axis=0)
    c_fear_avg_val = c_fear_val.mean(axis=0)
    c_joy_avg_val = c_joy_val.mean(axis=0)
    c_love_avg_val = c_love_val.mean(axis=0)
    c_sadness_avg_val = c_sadness_val.mean(axis=0)
    c_surprise_avg_val = c_surprise_val.mean(axis=0)

    # c_anger_avg_test = c_anger_test.mean(axis=0)
    # c_fear_avg_test = c_fear_test.mean(axis=0)


    visualize_c_heatmap(c_anger_avg_val[:, :192], 'anger',scale_str=scale_str)
    visualize_c_heatmap(c_fear_avg_val[:, :192], 'fear', scale_str=scale_str)
    visualize_c_heatmap(c_joy_avg_val[:, :192], 'joy', scale_str=scale_str)
    visualize_c_heatmap(c_love_avg_val[:, :192], 'love', scale_str=scale_str)
    visualize_c_heatmap(c_sadness_avg_val[:, :192], 'sadness', scale_str=scale_str)
    visualize_c_heatmap(c_surprise_avg_val[:, :192], 'surprise', scale_str=scale_str)


    # c_diff = abs(c_anger_avg_val - c_fear_avg_val)
    # visualize_c_heatmap(c_diff[:, :192], 'diff', scale_str=scale_str)

    unique_top_k_ang = []
    unique_top_k_fear = []
    k = 20
    top_k_ang = np.argsort(c_anger_avg_val.flatten())[-k:]
    top_k_fear = np.argsort(c_fear_avg_val.flatten())[-k:]
    for i in range(k):
        if not top_k_ang[i] in top_k_fear:
            unique_top_k_ang.append(top_k_ang[i])
        if not top_k_fear[i] in top_k_ang:
            unique_top_k_fear.append(top_k_fear[i])

    n_anger_val, I, H = c_anger_val.shape
    n_fear_val = c_fear_val.shape[0]

    # best_row_p, best_row_s, best_row_p_i, best_row_s_i = correlation_between_cs2(c_anger_val, c_fear_val)
    a=5

def correlation_between_cs(c1, c2, same=False):
    I = c1.shape[0]
    J = c2.shape[0]
    pearson_corrs = []
    spearman_corrs = []

    for i in range(I):
        for j in range(J):
            if same and i == j:
                continue
            p, _ = pearsonr(c1[i], c2[j])
            pearson_corrs.append(p)
            s, _ = spearmanr(c1[i], c2[j])
            spearman_corrs.append(s)
    return np.mean(pearson_corrs), np.mean(spearman_corrs)

def correlation_between_cs2(c1, c2):
    N1, I, H = c1.shape
    N2 = c2.shape[0]
    best_row_p = -1
    best_row_s = -1
    best_row_p_i = None
    best_row_s_i = None
    for i in tqdm(range(I)):
        pearson_corrs = []
        spearman_corrs = []
        for n1 in range(N1):
            for n2 in range(N2):
                if n1 == n2:
                    continue
                p_diff, _ = pearsonr(c1[n1, i, :], c2[n2, i, :])
                p_same, _ = pearsonr(c1[n1, i, :], c1[n2, i, :])
                pearson_corrs.append(np.abs(p_same) - np.abs(p_diff))
                s_diff, _ = spearmanr(c1[n1, i, :], c2[n2, i, :])
                s_same, _ = spearmanr(c1[n1, i, :], c1[n2, i, :])
                spearman_corrs.append(np.abs(s_same) - np.abs(s_diff))
        mean_row_pearson = np.mean(pearson_corrs)
        mean_row_spearman = np.mean(spearman_corrs)
        if mean_row_pearson > best_row_p:
            best_row_p = mean_row_pearson
            best_row_p_i = i
        if mean_row_spearman > best_row_s:
            best_row_s = mean_row_spearman
            best_row_s_i = i
    return best_row_p, best_row_s, best_row_p_i, best_row_s_i



def calc_XGBOOST_feats(relevant_c, counter_TH=0.015):
    I, H = relevant_c.shape
    # feature_vector = relevant_c.sum(axis=0) * np.sum(relevant_c > counter_TH, axis=0) / H
    feature_vector = np.concatenate([relevant_c.mean(axis=0), np.sum(relevant_c > counter_TH, axis=0) / H])
    return feature_vector


def train_XGBOOST(dataloader, model, device, relevant_neurons_indexes, wanted_labels=['anger','fear']):
    training_sentiments = []
    feature_matrix = []
    for idx, layer in tqdm(enumerate(dataloader), desc="Loading feats data", ascii=True):
        activation, label = layer
        sentiment = wanted_labels[label[0].item()]
        training_sentiments.append(label[0].item())
        activation = (activation - activation.mean(dim=1, keepdim=True)) / (activation.std(dim=1, keepdim=True) + 1e-5)
        activation = activation.clamp(-5, 5)  # Clip extreme values to prevent instability
        activation = activation.to(device)
        _, c, _ = model(activation)
        c = c.detach().cpu().numpy()
        relevant_c = c[relevant_neurons_indexes]
        feature_vector = calc_XGBOOST_feats(relevant_c)
        feature_matrix.append(feature_vector)
    X = np.array(feature_matrix)
    y = np.array(training_sentiments)
    clf = XGBClassifier(eval_metric='logloss', n_jobs=-1)
    clf.fit(X, y)

    return clf, X, y

def test_XGBOOST(dataloader, model, device, relevant_neurons_indexes, clf, wanted_labels=['anger','fear']):
    feature_matrix = []
    true_labels = []
    for idx, layer in tqdm(enumerate(dataloader), desc="Testing classifier", ascii=True):
        activation, label = layer
        sentiment = wanted_labels[label[0].item()]
        true_labels.append(label[0].item())

        activation = (activation - activation.mean(dim=1, keepdim=True)) / (activation.std(dim=1, keepdim=True) + 1e-5)
        activation = activation.clamp(-5, 5)  # Clip extreme values to prevent instability
        activation = activation.to(device)
        _, c, _ = model(activation)
        c = c.detach().cpu().numpy()
        relevant_c = c[relevant_neurons_indexes]
        feature_vector = calc_XGBOOST_feats(relevant_c)
        feature_matrix.append(feature_vector)

    X_test = np.array(feature_matrix)
    y_true = np.array(true_labels)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    return acc, cm, y_true, y_pred

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
