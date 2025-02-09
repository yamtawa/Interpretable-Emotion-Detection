import torch

def extract_layer_activation(model, input_ids, attention_mask, layer_index):
    with torch.no_grad():
        _, hidden_states = model(input_ids, attention_mask)
    return hidden_states[layer_index]  # Returns the activation map of the specified layer
