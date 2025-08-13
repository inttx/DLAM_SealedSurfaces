import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def train_loop(train_loader, val_loader, model, loss_fn, optimizer, num_epochs, device, save_path, model_type: str,
               plot_path, patch_size, batch_size, num_classes, patience=5):
    num_batches = len(train_loader)

    train_losses = []
    val_losses = []

    best_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", leave=False)
        for X, y in train_bar:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass
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
                pred = pred.view(batch_size, num_classes, patch_size, patch_size)
                loss = loss_fn(pred, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1} / {num_epochs}: Train Loss = {avg_epoch_loss:.4f}")

        # Validation
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
                    pred = pred.view(batch_size, num_classes, patch_size, patch_size)
                    loss = loss_fn(pred, y)

                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}: Validation Loss = {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"Validation improved — model saved to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break

    # Plot losses after training finishes
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
