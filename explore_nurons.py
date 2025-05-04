import torch
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import scipy.stats
from utils import get_wanted_label,get_dict_labels
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from visualizations import visualize_c_heatmap, visualize_F_heatmap
import os
from xgboost import XGBClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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

def loop_batches(model, dataloaders, device, criterion, function_name='extract_neuron_activations',wanted_labels='all', save_flag=True,
                 save_dir='data'):
    exp_func = get_function_from_name(function_name)
    keys,wanted_labels = get_wanted_label(wanted_labels)  # keys is a list of keys
    values = [dataloaders[key] for key in keys if key in dataloaders]
    neurons_all_activations = {}
    neurons_all_gradients = {}
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

            if device=='cuda':
                # Free up memory after each batch
                del data, attention_mask, target, logits, hidden_states
                torch.cuda.empty_cache()  # Clear CUDA cache
        neurons_all_activations[wanted_labels[idx]] = torch.stack(list(neuron_label_activation.values()))
        neurons_all_gradients[wanted_labels[idx]] = torch.stack(list(neuron_label_gradients.values()))
        if save_flag:
            for layer_idx, layer in enumerate(hidden_states):
                os.makedirs(save_dir, exist_ok=True)
                filename = 'activations_grads_' + wanted_labels[idx] + f'_layer{layer_idx}.pt'
                save_dict = {
                    "activations": neurons_all_activations[wanted_labels[idx]][0].unsqueeze(0),
                    # Dictionary of {label: tensor}, saving 0 layer only
                    "gradients": neurons_all_gradients[wanted_labels[idx]][0].unsqueeze(0)
                    # Dictionary of {label: tensor}, saving 0 layer only
                }
                torch.save(save_dict, os.path.join(save_dir, filename))
                print(
                    f'Saved {len(dataloader.dataset)} activations and grads for {wanted_labels[idx]} to file: {os.path.join(save_dir, filename)}')
    # d_labels=get_dict_labels(neurons_all_activations,keys,wanted_labels)
    # exp_func(neurons_all_activations,d_labels)



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
def extract_SAE_info(data_loader, type, model, device, wanted_labels, model_augmented=None, layer_idx='', scale_str='', top_T_vals=None):


    # Store lists of C matrices by sentiment label
    c_by_sentiment = defaultdict(list)
    aug_x_diff = {sentiment: 0 for sentiment in wanted_labels}

    for idx, layer in tqdm(enumerate(data_loader), desc=f"Extracting {type} info", ascii=True):
        activation, label = layer
        sentiment = wanted_labels[label[0].item()]

        activation = (activation - activation.mean(dim=1, keepdim=True)) / (activation.std(dim=1, keepdim=True) + 1e-5)
        activation = activation.clamp(-5, 5)
        activation = activation.to(device)

        x_hat, c, F_matrix = model(activation)
        if model_augmented:
            x_hat_aug, _, _ = model_augmented(activation)
            aug_x_diff[sentiment] += abs(x_hat_aug - x_hat).mean()
            asdasd = np.argsort(abs(x_hat_aug - x_hat).detach().cpu().numpy().squeeze(0))[-5:]
        c = c.detach().cpu().numpy()

        c_by_sentiment[sentiment].append(c)

    for sentiment in c_by_sentiment:
        c_by_sentiment[sentiment] = np.stack(c_by_sentiment[sentiment], axis=0)
        if model_augmented:
            aug_x_diff[sentiment] = (aug_x_diff[sentiment] / c_by_sentiment[sentiment].shape[0]).detach().cpu().numpy()
    # visualize_F_heatmap(F_matrix[:25, :50], layer_idx=layer_idx, scale_str=scale_str,
    #                     save_path=f'figures/F_mat_alpha05_scale{scale_str}layer{layer_idx}.png')

    return c_by_sentiment

def predict_layer_activation(model, val_dataloader, test_dataloader, device,
                             wanted_labels=[ 'anger', 'fear', 'joy', 'love', 'sadness', 'surprise'] ,
                             N=128, layer_idx=0, scale_str='2', alpha_str='025'):
    model.eval()
    keys, wanted_labels = get_wanted_label(wanted_labels)


    c_dict_val = extract_SAE_info(val_dataloader, type='val', model=model, device=device, wanted_labels=wanted_labels, layer_idx=layer_idx, scale_str=scale_str)


    c_class_avg_val = {sentiment: c_dict_val[sentiment].mean(axis=0) for sentiment in c_dict_val}

    for sentiment, c_avg in c_class_avg_val.items():
        visualize_c_heatmap(c_avg[:25, :50], sentiment, scale_str=scale_str)

    sentiment_c_diff = c_class_avg_val['anger'] - c_class_avg_val['joy']
    visualize_c_heatmap(sentiment_c_diff[:50, :50], 'diff', scale_str=scale_str, title='|C_avg_anger - C_avg_joy|')

    T = 10
    TH = 0.1

    top_T_val = np.argsort(sentiment_c_diff.mean(axis=0))[-T:]
    bigger_than_TH = np.sum(sentiment_c_diff > TH, axis=0)
    top_T_val = np.argsort(bigger_than_TH)[-T:]
    # model_augmented = copy.deepcopy(model)
    # with torch.no_grad():
    #     model_augmented.Decoder.weight[:, top_T_val] = 0

    c_dict_test = extract_SAE_info(test_dataloader, type='test', model=model, model_augmented=None, device=device,
                                   wanted_labels=wanted_labels, layer_idx=layer_idx, scale_str=scale_str, top_T_vals=top_T_val)


    corr_results = {}
    anchor_sentiment = 'anger'

    c_avg_rows_test = {sentiment: c_dict_test[sentiment].mean(axis=1) for sentiment in c_dict_test}
    for other_sentiment in c_avg_rows_test:
        same = other_sentiment == anchor_sentiment
        pearson_corr, spearman_corr = correlation_between_cs(
            c_avg_rows_test[anchor_sentiment],
            c_avg_rows_test[other_sentiment],
            same=same
        )
        corr_results[f"{anchor_sentiment} vs {other_sentiment}"] = {
            "pearson": pearson_corr,
            "spearman": spearman_corr
        }

    # Optional: print results
    for pair, corr_vals in corr_results.items():
        print(f"{pair}: Pearson = {corr_vals['pearson']:.4f}, Spearman = {corr_vals['spearman']:.4f}")

    clf = train_XGBOOST(c_dict_val, wanted_labels=wanted_labels, counter_TH=0.2)

    acc, f1 = test_XGBOOST(clf, c_dict_test, wanted_labels=wanted_labels, binary=False, counter_TH=0.2)
    print(f"Multi-class ACC: {acc:.3f}, F1: {f1:.3f}")

def correlation_between_cs(c1, c2, same=False):
    I = c1.shape[0]
    J = c2.shape[0]
    pearson_corrs = []
    spearman_corrs = []

    for i in tqdm(range(I)):
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



def calc_XGBOOST_feats(c_sample, counter_TH=0.1):
    I, H = c_sample.shape
    feature_vector = np.concatenate([c_sample.mean(axis=0), np.sum(c_sample > counter_TH, axis=0) / H])
    return feature_vector

def train_XGBOOST(c_dict, wanted_labels=['anger', 'fear'], counter_TH=0.1):
    X, y = [], []
    label_map = {label: i for i, label in enumerate(wanted_labels)}

    for label in wanted_labels:
        for c_sample in c_dict[label]:
            feat_vec = calc_XGBOOST_feats(c_sample, counter_TH)
            X.append(feat_vec)
            y.append(label_map[label])

    X, y = shuffle(np.array(X), np.array(y), random_state=42)
    clf = XGBClassifier(eval_metric='mlogloss')
    clf.fit(X, y)
    return clf

def test_XGBOOST(clf, c_dict, wanted_labels=['anger', 'fear'], counter_TH=0.1, binary=False):
    X_test, y_test = [], []
    label_map = {label: i for i, label in enumerate(wanted_labels)}

    for label in wanted_labels:
        for c_sample in c_dict[label]:
            feat_vec = calc_XGBOOST_feats(c_sample, counter_TH)
            X_test.append(feat_vec)
            y_test.append(label_map[label])

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_pred = clf.predict(X_test)

    if binary:
        f1 = f1_score(y_test, y_pred, average='binary')
        acc = accuracy_score(y_test, y_pred)
    else:
        f1 = f1_score(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)

    return acc, f1