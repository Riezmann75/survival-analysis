import torch
from lib.exception import StopTrainingError
import numpy as np
from torch import nn
from lib.metrics import c_index

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

def train_loop(dataloader, model, loss_fn, optimizer, required_grad=True):
    losses = []
    for _, (feature, label) in enumerate(dataloader):
        current_feat = feature.to(torch.float32)
        current_label = label.to(torch.float32)

        preds = model(current_feat)
        loss = loss_fn(preds, current_label[:, 0], current_label[:, 1])
        losses.append(loss.item())
        if torch.isinf(loss):
            raise StopTrainingError("Loss is Inf!")
        if torch.isnan(loss):
            raise StopTrainingError("Loss is NaN!")
        if required_grad:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return np.mean(losses)


def train_model_with_config(
    model: nn.Module,
    optimizer,
    num_epoch,
    scheduler,
    loss_fn,
    train_loader,
    validation_loader,
    test_loader,
):
    avg_losses = []
    val_losses = []
    for _ in range(num_epoch):
        model.train()

        epoch_loss = train_loop(train_loader, model, loss_fn, optimizer)
        avg_losses.append(epoch_loss)
        scheduler.step()

        # evaluate validation loss
        model.eval()
        with torch.no_grad():
            val_loss = train_loop(
                validation_loader,
                model,
                loss_fn,
                optimizer,
                required_grad=False,
            )
            val_losses.append(val_loss)

    # cindex on test set
    model.eval()
    with torch.no_grad():
        for feature, label in test_loader:
            X = feature.to(torch.float32)
            y = label.to(torch.float32)

            preds = model(X)
            c_index_value = c_index(preds, y[:, 0], y[:, 1])

    return avg_losses, val_losses, c_index_value
