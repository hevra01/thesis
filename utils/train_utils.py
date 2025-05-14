import torch

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        print(batch.keys())
        x = batch["images"].to(device)
        y = batch["labels"].to(device)
        print(batch["images"].shape)
        print(batch)
        exit()

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)
