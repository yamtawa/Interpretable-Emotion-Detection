import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc, balanced_accuracy_score, f1_score
import numpy as np
from sklearn.preprocessing import label_binarize, StandardScaler
from itertools import cycle
from load_config import load_config
import os
from load_config import load_label_map
import gzip
import joblib
from xgboost import XGBClassifier

from tqdm import tqdm
from utils import get_wanted_label
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


def get_activation_grads_path(label: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, f'activations_grads_{label}.gz')
    return file_path


if __name__ == '__main__':
    reduce_dim_method = ['grads', 'mean_0'][0]

    layer_num = [0, 5][0]
    config = load_config(config_name='config')
    label_map = load_label_map("dair-ai/emotion")
    file_name = 'all_activations.gz'

    wanted_labels_idx, wanted_labels = get_wanted_label(config['PIPLINE']['step2']['WANTED_LABELS'], load_config()["DATASET_PARAMS"]["DATASET_NAME"])
    if os.path.exists(file_name):
        with gzip.open(file_name, 'rb') as f:
            all_activations = joblib.load(f)
    else:
        all_activations = {}
        for label in tqdm(wanted_labels):
            with gzip.open(get_activation_grads_path(label), 'rb') as f:
                # current_file = joblib.load(f)
                all_activations[label] = joblib.load(f)
            # if layer_num == 1:
            # os.remove(get_activation_grads_path(label))

        with gzip.open(file_name, 'wb') as f:
            joblib.dump(all_activations, f)

    if reduce_dim_method == 'grads':
        # select best features based on the gradients maps
        top_k_features = {}
        k = 100
        for label in all_activations.keys():
            gradients = all_activations[label]['gradients'][layer_num, ...]
            gradients = gradients.numpy()
            tokenwise_importance = np.abs(gradients).sum(axis=1)
            tokenwise_importance = tokenwise_importance.mean(axis=0)
            top_k_features[label] = np.argsort(tokenwise_importance)[-k:]

        # intersect the top k features
        common = []
        union = []
        common_features = set(top_k_features[wanted_labels[0]])
        for label in wanted_labels[1:]:
            common_features = common_features.intersection(set(top_k_features[label]))
        common_features = list(common_features)
        print(f"n Common features: {len(common_features)}")

        # union the top k features
        union_features = set(top_k_features[wanted_labels[0]])
        for label in wanted_labels[1:]:
            union_features = union_features.union(set(top_k_features[label]))
        union_features = list(union_features)
        print(f"n Union features: {len(union_features)}")
        print(f"Union features: {union_features}")

        # features_dict = {}
        # for label in wanted_labels:
        #     for feature in top_k_features[label]:
        #         features_dict[feature] = features_dict.get(feature, 0) + 1

        activations = torch.cat([all_activations[label]['activations'][layer_num, ...] for label in all_activations.keys()], dim=0)
        selected_features_activations = activations[:, :, union_features]

        # pool across tokens
        X = selected_features_activations.mean(axis=1)

    elif reduce_dim_method == 'mean_0':
        X = torch.cat([all_activations[label]['activations'][layer_num, ...] for label in all_activations.keys()], dim=0)
        X = X.mean(dim=1)

    labels = [label_map[label] for label in all_activations.keys() for _ in range(all_activations[label]['activations'].shape[0])]
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBClassifier(objective='multi:softmax', num_class=6, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1_score = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"Balanced Accuracy: {balanced_accuracy}")
    print(f"F1 Score: {f1_score}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # plot ROC curve
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob, multi_class='ovr')}")
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5])
    n_classes = y_test_bin.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 'orange'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_{layer_num}_{reduce_dim_method}.png')





