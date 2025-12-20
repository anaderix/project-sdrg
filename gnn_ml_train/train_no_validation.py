# train.py  (FIXED, SAFE VERSION)

import torch
import torch.nn as nn

from pyg_dataset import SDRGDataset
from model import SDRGNet


EPOCHS = 20
LR = 1e-3


def train():
    dataset = SDRGDataset(root="data")

    sample = dataset[0]
    model = SDRGNet(
        node_dim=sample.x.shape[1],
        edge_dim=sample.edge_attr.shape[1],
        hidden_dim=64
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0

        for data in dataset:   # <-- NO batching
            optimizer.zero_grad()

            scores = model(data).unsqueeze(0)  # [1, num_edges]
            target = data.y.unsqueeze(0)       # [1]

            loss = criterion(scores, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = scores.argmax(dim=1)
            correct += (pred == target).sum().item()

        acc = correct / len(dataset)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Loss: {total_loss:.4f} | "
            f"Accuracy: {acc:.3f}"
        )


if __name__ == "__main__":
    train()

