# test_dataset.py

from pyg_dataset import SDRGDataset

dataset = SDRGDataset(root="data")

print("Number of samples:", len(dataset))

data = dataset[0]
print(data)

print("Node features shape:", data.x.shape)
print("Edge index shape:", data.edge_index.shape)
print("Edge features shape:", data.edge_attr.shape)
print("Target edge:", data.y.item())

