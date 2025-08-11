import torch


def train_loop(train_loader, val_loader, model, loss_fn, optimizer, num_epochs, device, save_path):
    num_batches = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        print(f"Starting epoch {epoch + 1} / {num_epochs}")
        for batch, (X, y) in enumerate(train_loader):
            print(f"Batch {batch + 1} / {num_batches}")
            X = X.to(device)
            y = y.to(device)

            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y.reshape((y.shape[0], pred.shape[-1])))

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item() / len(X)

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} / {num_epochs}: Average loss = {avg_epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch, (X, y) in enumerate(val_loader):
                X = X.to(device)
                y = y.to(device)

                outputs = model(X)

                # Initialize batch loss for this iteration
                batch_loss = 0.0
                for output, y in zip(output, y):
                    # Compute loss for each pair
                    loss = loss_fn(outputs, y)
                    batch_loss += loss  # Accumulate as tensor

                # Average the loss over the batch (keep it as tensor)
                val_loss += batch_loss.item() / len(X)

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}!")
