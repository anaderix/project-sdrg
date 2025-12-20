# train_with_validation.py

import os
import torch
import torch.nn as nn

from pyg_dataset import SDRGDataset
from model import SDRGNet
from split_dataset import split_indices
from evaluate import evaluate
from checkpoint import save_checkpoint, load_checkpoint


# -------------------------
# Hyperparameters
# -------------------------
EPOCHS = 50
LR = 1e-3
CHECKPOINT_PATH = "checkpoint.pt"
RESUME = True


def train():
    dataset = SDRGDataset(root="data")

    train_idx, val_idx, test_idx = split_indices(len(dataset))

    sample = dataset[0]
    model = SDRGNet(
        node_dim=sample.x.shape[1],
        edge_dim=sample.edge_attr.shape[1],
        hidden_dim=64
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val_acc = 0.0

    # -------------------------
    # Resume if checkpoint exists
    # -------------------------
    if RESUME and os.path.exists(CHECKPOINT_PATH):
        start_epoch, best_val_acc = load_checkpoint(
            CHECKPOINT_PATH, model, optimizer
        )
        print(f"Resumed from epoch {start_epoch}, best val acc = {best_val_acc:.3f}")

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0

        for idx in train_idx:
            data = dataset[idx]

            optimizer.zero_grad()
            scores = model(data).unsqueeze(0)
            target = data.y.unsqueeze(0)

            loss = criterion(scores, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() / len(train_idx)


        val_acc = evaluate(model, dataset, val_idx)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train loss: {total_loss:.4f} | "
            f"Val acc: {val_acc:.3f}"
        )

        # -------------------------
        # Save best model
        # -------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                CHECKPOINT_PATH,
                model,
                optimizer,
                epoch + 1,
                best_val_acc
            )
            print("  ✓ Best model saved")

    # -------------------------
    # Final test evaluation
    # -------------------------
    load_checkpoint(CHECKPOINT_PATH, model)
    test_acc = evaluate(model, dataset, test_idx)

    print(f"\nFinal TEST accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    train()

