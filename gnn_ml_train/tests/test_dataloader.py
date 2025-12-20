# test_dataloader.py

from torch_geometric.loader import DataLoader
from pyg_dataset import SDRGDataset

dataset = SDRGDataset(root="data")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in loader:
    print(batch)
    print("Batch edge_attr shape:", batch.edge_attr.shape)
    print("Batch y:", batch.y)
    break

