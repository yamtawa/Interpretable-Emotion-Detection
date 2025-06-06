import torch

def extract_layer_activation(model, input_ids, attention_mask, layer_index):
    with torch.no_grad():
        _, hidden_states = model(input_ids, attention_mask)
    return hidden_states[layer_index]  # Returns the activation map of the specified layer


def get_wanted_label(wanted_labels):
    label_maping={"anger": 0, "fear": 1, "joy": 2, "love": 3, "sadness": 4, "surprise": 5}
    if wanted_labels == "all":
        return list(label_maping.values()),list(label_maping.keys())
    return [label_maping[wr_label] for wr_label in wanted_labels],wanted_labels

def get_dict_labels(neurons_all_activations, keys, wanted_labels):
    return {
        label_wr: torch.ones(activation.shape[1]) * wanted_labels[keys.index(label_wr)]
        for label_wr, activation in neurons_all_activations.items()
        if label_wr in keys
    }