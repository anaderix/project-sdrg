# test_model.py

import torch
from pyg_dataset import SDRGDataset
from model import SDRGNet

dataset = SDRGDataset(root="data")
data = dataset[0]

model = SDRGNet(
    node_dim=data.x.shape[1],
    edge_dim=data.edge_attr.shape[1]
)

scores = model(data)

print("Scores shape:", scores.shape)
print("Number of edges:", data.edge_attr.shape[0])
print("Predicted edge:", torch.argmax(scores).item())
print("Target edge:", data.y.item())

