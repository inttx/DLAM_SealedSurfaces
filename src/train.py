import torch
from tqdm import tqdm


def train_loop(train_loader, val_loader, model, loss_fn, optimizer, num_epochs, device, save_path):
    num_batches = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        train_bar = tqdm(train_loader, desc="Training", leave=False)
        for X, y in train_bar:
            X = X.to(device)
            y = y.to(device)

            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y.reshape((y.shape[0], pred.shape[-1])))

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} / {num_epochs}: Average loss = {avg_epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for X, y in val_bar:
                X = X.to(device)
                y = y.to(device)
                outputs = model(X)
                loss = loss_fn(outputs, y)
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} average validation loss: {avg_val_loss:.4f}")


    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}!")
