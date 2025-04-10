def predict_layer_activation(model, val_dataloader, test_dataloader, device, wanted_labels=['anger','fear'], N=128, layer_idx=0, scale_str='2', alpha_str='025'):
    model.eval()
    keys, wanted_labels = get_wanted_label(wanted_labels)
    c_fear_sum = 0
    fear_counter = 0
    c_anger_sum = 0
    anger_counter = 0
    TH = 2

    sentence_means = []
    sentence_maxes = []
    sentence_rates = []
    sentence_labels = []
    for idx, layer in tqdm(enumerate(val_dataloader), desc="Testing progress", ascii=True):
        activation, label = layer
        sentiment = wanted_labels[label[0].item()]
        activation = (activation - activation.mean(dim=1, keepdim=True)) / (activation.std(dim=1, keepdim=True) + 1e-5)
        activation = activation.clamp(-5, 5)  # Clip extreme values to prevent instability
        activation = activation.to(device)
        x_hat, c, F_matrix = model(activation)
        x_hat, c, F_matrix = x_hat.detach().cpu().numpy(), c.detach().cpu().numpy(), F_matrix.detach().cpu().numpy()
        # visualize_c_heatmap(c, sentiment, scale_str=scale_str)
        I, H = c.shape
        c_sentence_mean = np.mean(c, axis=0)  # shape: [num_features]
        c_sentence_max = np.max(c, axis=0) / I
        c_higher_than_TH = np.mean(c > TH, axis=0)

        sentence_means.append(c_sentence_mean)
        sentence_maxes.append(c_sentence_max)
        sentence_rates.append(c_higher_than_TH)
        sentence_labels.append(label[0].item())

    X = np.stack(sentence_means)  # shape [num_sentences, num_features]
    y = np.array(sentence_labels)  # 0 = fear, 1 = anger
    model = XGBClassifier(
        objective='binary:logistic',
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X, y)
    importances = model.feature_importances_
    top_feats = np.argsort(importances)[::-1][:10]

    # visualize_F_heatmap(F_matrix[:20, :20], sentiment)
    # c_anger_avg = c_anger_sum / anger_counter
    # c_fear_avg = c_fear_sum / fear_counter
    # c_diff = abs(c_anger_avg - c_fear_avg)
    # c_diff_per_feature = c_diff.sum(axis=0)
    # # visualize_c_heatmap(c_anger_avg[:, :192], 'anger', save_path=f'figures/anger_avg_alpha{alpha_str}_layer{layer_idx}', scale_str=scale_str)
    # visualize_c_heatmap(c_fear_avg[:, :192], 'fear', save_path=f'figures/fear_avg_alpha{alpha_str}_layer{layer_idx}', scale_str=scale_str)
    # visualize_c_heatmap(c_diff[:, :192], 'diff', save_path=f'figures/diff{alpha_str}_layer{layer_idx}',
    #                     scale_str=scale_str)


    # flat_c_diff = c_diff.flatten()
    # k = 10#int(0.001 * len(flat_c_diff))
    # top_k_indices = np.argpartition(flat_c_diff, -k)[-k:]
    # top_rows, top_cols = np.unravel_index(top_k_indices, c_diff.shape)

    # neuron_TH = 0.05
    # relevant_neurons_indexes = np.where(c_diff.mean(axis=1) > neuron_TH)[0]
    # relevant_c_diff = c_diff[relevant_neurons_indexes]
    # visualize_c_heatmap(relevant_c_diff[:, :192], 'diff', save_path=f'figures/relevant_diff{alpha_str}_layer{layer_idx}',
    #                     scale_str=scale_str)
    #
    # clf, X, y = train_XGBOOST(val_dataloader, model, device, relevant_neurons_indexes, wanted_labels)
    #
    # acc, cm, y_true, y_pred = test_XGBOOST(test_dataloader, model, device, relevant_neurons_indexes, clf, wanted_labels=['anger','fear'])
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