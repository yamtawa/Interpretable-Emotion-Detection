import os
from tqdm import tqdm
from weights_loader import *

import matplotlib.pyplot as plt

# batch = next(iter(dataloader))
# input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]

def training_loop(model,train_loader,test_loader,optimizer,criterion,device,best_val_loss,current_config):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=current_config['CURRENT_STEP']['NUM_EPOCHS'], eta_min=1e-6)
    val_loss_l ,train_loss_l=[],[]
    val_accu_l =[]
    for epoch in range(current_config['CURRENT_STEP']['NUM_EPOCHS']):
        print(f"\nEpoch {epoch+1} out of {current_config['CURRENT_STEP']['NUM_EPOCHS']}")
        train_loss,model= train(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy,_  = eval(model, test_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if current_config['GENERAL']['SAVED_MODEL_NAME'][0]:
                save_best_weights(model, optimizer, epoch, best_val_loss,os.path.join(os.getcwd(),'models_weights',f"{current_config['GENERAL']['SAVED_MODEL_NAME'][1]}.pth"))
        print(f"\nEpoch {epoch + 1}/{current_config['CURRENT_STEP']['NUM_EPOCHS']}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss} \n"
              f"Val Accuracy: {val_accuracy}")
        scheduler.step()
        val_loss_l.append(val_loss),train_loss_l.append(train_loss),val_accu_l.append(val_accuracy)
    return model,best_val_loss


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc="Batch Progress", ascii=True):
        data = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["labels"].to(device)
        optimizer.zero_grad()
        logits, all_hidden_states = model(data, attention_mask)
        loss = criterion(logits, target.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if device == 'cuda':
            # Free up memory after each batch
            del data, attention_mask, target, logits
            torch.cuda.empty_cache()
    return running_loss / len(train_loader), model



def eval(model, loader, criterion, device):
    running_loss,correct,total = 0.,0.,0.
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader,desc="Batch Progress",ascii=True):
            data = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["labels"].to(device)
            logits, all_hidden_states = model(data,attention_mask)
            loss = criterion(logits, target.long())
            running_loss += loss.item()
            predictions = torch.argmax(logits, dim=1) # TODO- MAKE THIS ADAPTABLE TO MULTILABEL CLASSIFICATION ALSO
            correct += (predictions == target).sum().item()
            total += target.size(0)
            if device == 'cuda':
                # Free up memory after each batch
                del data, attention_mask, target, logits
                torch.cuda.empty_cache()

        accuracy = correct / total if total > 0 else 0
    return running_loss / len(loader),accuracy,model





def train_SAE(model, train_loader, optimizer, criterion, device, current_config):
    model.train()
    running_loss = 0.0
    running_loss_recon = 0.0
    running_loss_sparsity = 0.0

    for batch in tqdm(train_loader, desc="SAE Batch Progress",ascii=True):
        optimizer.zero_grad()

        activation, label = batch
        if not current_config['NEURON_DATASET_PARAMS']['ROW_NEURONS']:
            activation = activation.flatten(start_dim=1)
        activation = (activation - activation.mean(dim=1, keepdim=True)) / (activation.std(dim=1, keepdim=True) + 1e-5)
        activation = activation.clamp(-5, 5)  # Clip extreme values to prevent instability
        activation = activation.to(device)
        x_hat, c, _ = model(activation)
        loss, loss_recon, loss_sparsity = criterion(x_hat, activation, c)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_loss_recon += loss_recon
        running_loss_sparsity += loss_sparsity

        if device == "cuda":
            torch.cuda.empty_cache()
    return running_loss / len(train_loader), running_loss_recon / len(train_loader), running_loss_sparsity / len(train_loader), model



def eval_SAE(model, loader, criterion, device, current_config):
    model.eval()
    running_loss = 0.0
    running_loss_recon = 0.0
    running_loss_sparsity = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating SAE", ascii=True):
            activation, label = batch  # Only activations needed
            if not current_config['NEURON_DATASET_PARAMS']['ROW_NEURONS']:
                activation = activation.flatten(start_dim=1)
            activation = (activation - activation.mean(dim=1, keepdim=True)) / (
                        activation.std(dim=1, keepdim=True) + 1e-5)
            activation = activation.clamp(-5, 5)  # Clip extreme values to prevent instability
            activation = activation.to(device)

            # Forward pass
            x_hat, c, _ = model(activation)

            # Compute loss (L2 + L1)
            loss, loss_recon, loss_sparsity = criterion(x_hat, activation, c)
            running_loss += loss.item()
            running_loss_recon += loss_recon.item()
            running_loss_sparsity += loss_sparsity.item()

            if device == "cuda":
                torch.cuda.empty_cache()

    return running_loss / len(loader), running_loss_recon / len(loader), running_loss_sparsity / len(loader), model

def training_loop_SAE(model, train_loader, test_loader, optimizer, criterion, device, best_val_loss, current_config, plot_loss=True, save_weights_every=None):
    num_epochs = current_config['CURRENT_STEP']["NUM_EPOCHS"]
    scale = str(current_config['CURRENT_STEP']['OUT_SCALE']).replace(".", "")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-6)
    val_loss_l, train_loss_l = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1} out of {num_epochs}")

        # Training Step
        train_loss, train_recon_loss, train_sparsity_loss, model = train_SAE(model, train_loader, optimizer, criterion, device, current_config)

        # Validation Step
        val_loss, val_recon_loss, val_sparsity_loss, _ = eval_SAE(model, test_loader, criterion, device, current_config)

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_weights_every:
                save_best_weights(model, optimizer, epoch, best_val_loss,
                                  os.path.join(os.getcwd(), 'models_weights',
                                               f"SAE_epoch{epoch}_{scale}.pth"))
        # Save every `save_weights_every` epochs
        # if epoch % save_weights_every == 0:
        #     save_path = os.path.join(os.getcwd(), 'models_weights',
        #                              f"{current_config['GENERAL']['SAVED_MODEL_NAME'][1]}_SAE_epoch{epoch}.pth")
        #     save_best_weights(model, optimizer, epoch, val_loss, save_path)
        #     print(f"📌 Model checkpoint saved at epoch {epoch}")
        print(f"\nEpoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Recon Loss: {train_recon_loss:.4f}, Train Sparsity Loss: {train_sparsity_loss:.4f}\n"
              f"Val Loss: {val_loss:.4f}, Val Recon Loss: {val_recon_loss:.4f}, Val Sparsity Loss: {val_sparsity_loss:.4f}")

        val_loss_l.append(val_loss)
        train_loss_l.append(train_loss)
    # Save the final model at the end of training
    final_model_path = os.path.join(os.getcwd(), 'models_weights',
                                    f"SAE_try1_final_{scale}_layer{current_config['NEURON_DATASET_PARAMS']['LAYER_IDX']}.pth")
    save_best_weights(model, optimizer, num_epochs, val_loss, final_model_path)
    print(f"✅ Final model saved at epoch {num_epochs}")
    if plot_loss:
        # Plot the loss curves
        plot_loss_curve(train_loss_l, val_loss_l, save_path=os.path.join(os.getcwd(), 'figures', f"loss_plot_SAE_{scale}_layer{current_config['NEURON_DATASET_PARAMS']['LAYER_IDX']}.png"))

    return model, best_val_loss


def plot_loss_curve(train_losses, val_losses, save_path=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Loss plot saved to {save_path}")
    else:
        plt.show()
