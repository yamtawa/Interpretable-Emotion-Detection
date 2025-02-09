import torch


def save_best_weights(model, optimizer, epoch, val_loss, save_path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(state, save_path)
    print(f"Best weights saved to {save_path}")


def load_best_weights(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    print(f"Loaded best weights from {load_path}, Epoch: {epoch}, Validation Loss: {val_loss}")
    return model, val_loss



