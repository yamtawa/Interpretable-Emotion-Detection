import os
from tqdm import tqdm
from weights_loader import *

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

        accuracy = correct / total if total > 0 else 0
    return running_loss / len(loader),accuracy,model




