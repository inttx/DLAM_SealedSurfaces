def train_loop(dataloader, model, loss_fn, optimizer, num_epochs, device):
    num_batches = len(dataloader)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch, (X, y) in enumerate(dataloader):
            print(f"Batch {batch+1} / {num_batches}")
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

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} / {num_epochs}: Average loss = {avg_epoch_loss:.4f}")
