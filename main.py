import os
import numpy as np
import torch
from data_prep import get_dataloader, get_neuron_dataloader
from models import EmotionBERT, SAE, SAELoss
from load_config import load_config
from training_page import training_loop, eval, training_loop_SAE, eval_SAE
from criterions_and_optimizers import set_optimizer, set_criterion
from weights_loader import load_best_weights
from explore_nurons import loop_batches, predict_layer_activation

def backbone_main():
    best_val_loss = np.inf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    dataloader_train,_ = get_dataloader(dataset_name=current_config["DATASET_PARAMS"]["DATASET_NAME"],
                                      batch_size=current_config["DATASET_PARAMS"]["BATCH_SIZE"], split='train',ratio_small=5)
    dataloader_eval, dataloader_test = get_dataloader(dataset_name=current_config["DATASET_PARAMS"]["DATASET_NAME"],
                                                      batch_size=current_config["DATASET_PARAMS"]["BATCH_SIZE"],
                                                      split='test',ratio_small=2)
    print(
        f"Train samples:{len(dataloader_train.dataset)}\nVal samples:{len(dataloader_train.dataset.dataset)}\nTest samples:{len(dataloader_test.dataset)}\n")
    model = EmotionBERT(model_name=current_config["MODELS_PARAMS"]["MODEL_NAME"],
                        num_labels=len(dataloader_train.dataset.dataset.label_map),
                        fine_tune_only=current_config["CURRENT_STEP"]["FINE_TUNE_ONLY"]).to(device)
    optimizer = set_optimizer(model, current_config)
    criterion = set_criterion(current_config)
    if current_config['CURRENT_STEP']['TRAINING_PHASE'] == "retrain" or current_config['CURRENT_STEP'][
        'TRAINING_PHASE'] == "eval":
        model, _ = load_best_weights(model, optimizer, os.path.join(os.getcwd(), 'models_weights',
                                                                    f"{current_config['GENERAL']['SAVED_MODEL_NAME'][1]}.pth"))
    if not current_config['CURRENT_STEP']['TRAINING_PHASE'][0] == "eval":
        model, best_val_loss = training_loop(model, dataloader_train, dataloader_eval, optimizer, criterion, device,
                                             best_val_loss, current_config)
    test_loss, test_accuracy, _ = eval(model, dataloader_test, criterion, device)
    print(f"Test average accuracy:{test_accuracy}")
    print(f"Test Loss: {np.array(test_loss)}, Test Accuracy: {test_accuracy}")
    # TODO- WORK ON ACTIVATIONS VISUALIZATIONS AND ADD TOOLS TO ANALYZE ACTIVATIONS

def get_activations_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, dataloader_explore= get_dataloader(dataset_name=current_config["DATASET_PARAMS"]["DATASET_NAME"],
                                      batch_size=current_config["DATASET_PARAMS"]["BATCH_SIZE"], split='train',ratio_small=5,labels_dl=True)
    model = EmotionBERT(model_name=current_config["MODELS_PARAMS"]["MODEL_NAME"],
                        num_labels=len(dataloader_explore),
                        fine_tune_only=current_config["CURRENT_STEP"]["FINE_TUNE_ONLY"]).to(device)
    optimizer = set_optimizer(model, current_config)
    criterion = set_criterion(current_config)
    weights_path=os.path.join(os.getcwd(), 'models_weights',f"{current_config['GENERAL']['SAVED_MODEL_NAME'][1]}.pth")
    if os.path.isfile(weights_path):
        model, _ = load_best_weights(model, optimizer, weights_path)
    loop_batches(model,dataloader_explore,device,criterion=criterion,wanted_labels=current_config['CURRENT_STEP']['WANTED_LABELS'], layer_index=current_config["NEURON_DATASET_PARAMS"]["LAYER_IDX"])

def train_SAE_main():
    best_val_loss = np.inf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Load Train & Validation DataLoaders
    dataloader_train, dataloader_eval, dataloader_test = get_neuron_dataloader(
        wanted_labels=current_config['CURRENT_STEP']['WANTED_LABELS'],
        layer_index=current_config["NEURON_DATASET_PARAMS"]["LAYER_IDX"],  # Example: select specific layers
        data_type="activations",
        batch_size=current_config["NEURON_DATASET_PARAMS"]["BATCH_SIZE"], N=current_config['CURRENT_STEP']['LANG_INPUT']
    )
    print(f"Train samples: {len(dataloader_train.dataset)}")
    print(f"Validation samples: {len(dataloader_eval.dataset)}")
    print(f"Test samples: {len(dataloader_test.dataset)}\n")

    # Initialize SAE Model
    sae = SAE(input_dim=current_config['CURRENT_STEP']['LANG_D'], d_feat_scale=current_config['CURRENT_STEP']['OUT_SCALE']).to(device)  # Assuming hidden size = 768

    # Define Optimizer & Custom SAE Loss
    optimizer = torch.optim.Adam(
        sae.parameters(),
        lr=current_config['CURRENT_STEP']['LR_SAE'],
        weight_decay=current_config['CURRENT_STEP']['WEIGHT_DECAY']
    )
    criterion = SAELoss(alpha_sparsity=current_config['CURRENT_STEP']['ALPHA_SPARSITY'])  # SAE loss (MSE + L1 sparsity)

    # Load best weights if retraining or evaluating

    sae, best_val_loss = training_loop_SAE(
        sae, dataloader_train, dataloader_eval, optimizer, criterion, device, best_val_loss, current_config
    )

def explore_neurons_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Load Train & Validation DataLoaders
    dataloader_train, dataloader_val, dataloader_test = get_neuron_dataloader(
        wanted_labels=current_config['CURRENT_STEP']['WANTED_LABELS'],
        layer_index=current_config['NEURON_DATASET_PARAMS']['LAYER_IDX'],  # Example: select specific layers
        data_type="activations",
        batch_size=current_config['CURRENT_STEP']['LANG_INPUT'], N=current_config['CURRENT_STEP']['LANG_INPUT']
    )

    print(f"Train samples: {len(dataloader_train.dataset)}")
    print(f"Validation samples: {len(dataloader_val.dataset)}")
    print(f"Test samples: {len(dataloader_test.dataset)}\n")


    # Initialize SAE Model
    model = SAE(input_dim=current_config['CURRENT_STEP']['LANG_D'],
              d_feat_scale=current_config['CURRENT_STEP']['OUT_SCALE']).to(device)  # Assuming hidden size = 768
    optimizer = torch.optim.Adam(model.parameters(), lr=current_config['CURRENT_STEP']['LR_SAE'],
        weight_decay=current_config['CURRENT_STEP']['WEIGHT_DECAY'])
    scale = str(current_config['CURRENT_STEP']['OUT_SCALE']).replace(".", "")
    alpha = str(current_config['CURRENT_STEP']['ALPHA_SPARSITY']).replace(".", "")
    model, optimizer, val_loss = load_best_weights(model, optimizer, os.path.join(os.getcwd(), 'models_weights', f"{current_config['CURRENT_STEP']['MODEL']}_final_scale{scale}_alpha{alpha}_layer{current_config['NEURON_DATASET_PARAMS']['LAYER_IDX']}.pth"))

    predict_layer_activation(model, dataloader_val, dataloader_test, device, layer_idx=current_config['NEURON_DATASET_PARAMS']['LAYER_IDX'], scale_str=scale, alpha_str=alpha,
                             wanted_labels=current_config['CURRENT_STEP']['WANTED_LABELS'])



if __name__ == "__main__":
    config = load_config(config_name='config')

    current_config = config.copy()
    current_config.pop('PIPLINE', None)
    current_config['CURRENT_STEP'] = {}


    for step_index, s in enumerate(config['PIPLINE'].values()):
        if not s['ACTIVATE']:
            continue

        current_config['CURRENT_STEP']['WANTED_LABELS'] = s['WANTED_LABELS']
        current_config['CURRENT_STEP']['WEIGHT_DECAY'] = s['WEIGHT_DECAY']

        if  s['TITLE']=="GET_ACTIVATIONS":
            current_config['CURRENT_STEP']['FINE_TUNE_ONLY'] = s['FINE_TUNE_ONLY']
            current_config['CURRENT_STEP']['OPTIMIZER_NAME'] = s['OPTIMIZER_NAME']
            current_config['CURRENT_STEP']['LR_FEATURES'] = s['LR_FEATURES']
            current_config['CURRENT_STEP']['LR_HEAD'] = s['LR_HEAD']

            current_config['CURRENT_STEP']['CRITERION_NAME'] = s['CRITERION_NAME']
            get_activations_main()

        elif s['TITLE']=="MODEL_TRAINING":
            current_config['CURRENT_STEP']['MODEL'] = s['MODEL']
            current_config['CURRENT_STEP']['FINE_TUNE_ONLY'] = s['FINE_TUNE_ONLY']
            current_config['CURRENT_STEP']['OPTIMIZER_NAME'] = s['OPTIMIZER_NAME']
            current_config['CURRENT_STEP']['TRAINING_PHASE'] = s['TRAINING_PHASE']
            current_config['CURRENT_STEP']['NUM_EPOCHS'] = s['NUM_EPOCHS']
            current_config['CURRENT_STEP']['SCHEDULER_NAME'] = s['SCHEDULER_NAME']
            current_config['CURRENT_STEP']['CRITERION_NAME'] = s['CRITERION_NAME']
            current_config['CURRENT_STEP']['ALPHA_SPARSITY'] = s['ALPHA_SPARSITY']
            current_config['CURRENT_STEP']['LR_SAE'] = s['LR_SAE']

            print(current_config['CURRENT_STEP'])
            if s['MODEL'] == 'BERT':
                backbone_main()
            else:
                current_config['CURRENT_STEP']['LANG_INPUT'] = s['LANG_INPUT']
                current_config['CURRENT_STEP']['LANG_D'] = s['LANG_D']
                current_config['CURRENT_STEP']['OUT_SCALE'] = s['OUT_SCALE']
                current_config['CURRENT_STEP']['LR_SAE'] = s['LR_SAE']

                train_SAE_main()

        else:
            current_config['CURRENT_STEP']['LANG_INPUT'] = s['LANG_INPUT']
            current_config['CURRENT_STEP']['LANG_D'] = s['LANG_D']
            current_config['CURRENT_STEP']['OUT_SCALE'] = s['OUT_SCALE']
            current_config['CURRENT_STEP']['ALPHA_SPARSITY'] = s['ALPHA_SPARSITY']
            current_config['CURRENT_STEP']['LR_SAE'] = s['LR_SAE']
            current_config['CURRENT_STEP']['MODEL'] = s['MODEL']
            explore_neurons_main()


