import torch
from tqdm import tqdm
import collections
import torch.nn.functional as F
import matplotlib.pyplot as plt


def train_loop(train_loader, val_loader, model, loss_fn, optimizer, num_epochs, device, save_path, model_type: str,
               plot_path):
    num_batches = len(train_loader)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        train_bar = tqdm(train_loader, desc="Training", leave=False)
        for X, y in train_bar:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Compute prediction and loss
            if model_type == 'SegFormer':
                pred = model(pixel_values=X).logits
            else:
                pred = model(X)
            if model_type == 'DeepLabV3':
                loss1 = loss_fn(pred['out'], y)
                loss2 = loss_fn(pred['aux'], y)
                loss = loss1 + 0.4 * loss2
            elif model_type == 'SegFormer':
                pred_upsampled = F.interpolate(pred, size=(y.shape[1], y.shape[2]), mode='bilinear',
                                               align_corners=False)
                loss = loss_fn(pred_upsampled, y)
            else:
                loss = loss_fn(pred, y.reshape((y.shape[0], pred.shape[-1])))

            # Backpropagation
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1} / {num_epochs}: Average loss = {avg_epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for X, y in val_bar:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if model_type == 'SegFormer':
                    pred = model(pixel_values=X).logits
                else:
                    pred = model(X)
                if model_type == 'DeepLabV3':
                    loss1 = loss_fn(pred['out'], y)
                    loss2 = loss_fn(pred['aux'], y)
                    loss = loss1 + 0.4 * loss2
                elif model_type == 'SegFormer':
                    pred_upsampled = F.interpolate(pred, size=(y.shape[1], y.shape[2]), mode='bilinear',
                                                   align_corners=False)
                    loss = loss_fn(pred_upsampled, y)
                else:
                    loss = loss_fn(pred, y.reshape((y.shape[0], pred.shape[-1])))
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1} average validation loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}!")

    # Plot losses after training finishes
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
