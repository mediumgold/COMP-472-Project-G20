"""
Feed-forward neural network (multi-layer perceptron) in PyTorch.

Architecture: input -> [Linear -> activation -> (dropout)] for each hidden layer -> Linear -> class scores
Training:     mini-batch gradient descent (Adam or SGD) on the cross-entropy loss.

After every epoch we record train/validation loss and accuracy and keep the weights
from the epoch with the best validation accuracy (early stopping).

hidden=() means no hidden layer, which is just multinomial logistic regression.
"""
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(256, 64), activation="relu", dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))   # raw scores; the softmax is inside the loss
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, X, y, loss_fn, batch=4096):
    model.eval()
    total_loss, correct = 0.0, 0
    for i in range(0, len(X), batch):
        out = model(X[i:i + batch])
        total_loss += loss_fn(out, y[i:i + batch]).item() * len(out)
        correct += (out.argmax(1) == y[i:i + batch]).sum().item()
    return total_loss / len(X), correct / len(X)


def train_mlp(X_train, y_train, X_val=None, y_val=None, n_classes=None,
              hidden=(256, 64), activation="relu", dropout=0.0, lr=1e-3, optimizer="adam",
              weight_decay=0.0, batch_size=128, epochs=50, seed=42, patience=None):
    """
    Train an MLP and return (model, history DataFrame, best_epoch).
    With validation data, the returned model has the weights of the best validation epoch,
    and `patience` stops training after that many epochs without improvement.
    Without validation data (the final model), we just train for `epochs`.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_classes = n_classes or int(np.max(y_train)) + 1
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    has_val = X_val is not None
    if has_val:
        Xv = torch.tensor(X_val, dtype=torch.float32)
        yv = torch.tensor(y_val, dtype=torch.long)

    model = MLP(Xt.shape[1], n_classes, hidden, activation, dropout)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    history = []
    best_acc, best_epoch, best_weights = -1.0, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)   # forward pass
            loss.backward()                 # backpropagation
            opt.step()                      # weight update

        train_loss, train_acc = evaluate(model, Xt, yt, loss_fn)
        row = dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc)
        if has_val:
            val_loss, val_acc = evaluate(model, Xv, yv, loss_fn)
            row.update(val_loss=val_loss, val_acc=val_acc)
            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epoch
                best_weights = copy.deepcopy(model.state_dict())
        history.append(row)
        if has_val and patience is not None and epoch - best_epoch >= patience:
            break

    if has_val:
        model.load_state_dict(best_weights)
    else:
        best_epoch = epochs
    return model, pd.DataFrame(history), best_epoch


@torch.no_grad()
def predict_proba(model, X, batch=4096):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32)
    out = [torch.softmax(model(X[i:i + batch]), dim=1) for i in range(0, len(X), batch)]
    return torch.cat(out).numpy()
