import json

import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import numpy as np
import joblib  # or pickle
from collections import Counter



def preform_kmeans(data,saved_path,K=6):
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(torch.concatenate(list(data.values())))
    cluster_centers = kmeans.cluster_centers_  # shape: (K, n_features)
    joblib.dump(cluster_centers.tolist(), saved_path+'_kmeans_centers.pkl')



def label_new_sample(data,saved_path):
    full_preds_l=[]
    final_preds_l=[]
    counter=0
    cluster_centers = joblib.load( saved_path+'_kmeans_centers.pkl')  # shape: (K, n_features)
    for X_test_dict in data:
        true_label=X_test_dict.pop('true_label')
        samp_dict={}
        for k,v in X_test_dict.items():
            v=np.array(v)
            v = v[np.newaxis, :] if v.ndim==1 else v
            distances = cdist(v, cluster_centers)  # shape: (n_test_samples, K)
            # labels = np.argmin(distances, axis=1)
            samp_dict[k]=distances[:,int(k)] #use label instead of k
        # value_counts = Counter(samp_dict.values())
        # most_common_value, count = value_counts.most_common(1)[0]
        pred_value=int(min(samp_dict, key=samp_dict.get))
        samp_dict['true_label']=true_label
        final_preds_l.append({'gt':true_label,'pred': pred_value})
        full_preds_l.append(samp_dict)
        counter+=true_label==pred_value
    print(f"Accuracy: {counter/len(data)}")



if __name__ == '__main__':
    with open('selected_grads.json', 'r') as f:
        wanted_neuorns = json.load(f)
    label_new_sample(wanted_neuorns, 'one_mask_for_all_grads_only_block-1_1000activated_neurons')