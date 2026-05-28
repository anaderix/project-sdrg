# pyg_dataset.py

import os
import json
import torch
from torch_geometric.data import Dataset, Data


class SDRGDataset(Dataset):
    def __init__(self, root="data"):
        super().__init__()
        self.root = root
        self.samples = self._collect_files()

    def _collect_files(self):
        files = []
        for realization in sorted(os.listdir(self.root)):
            rdir = os.path.join(self.root, realization)
            if not os.path.isdir(rdir):
                continue
            for fname in sorted(os.listdir(rdir)):
                if fname.endswith(".json"):
                    files.append(os.path.join(rdir, fname))
        return files

    def len(self):
        return len(self.samples)

    def get(self, idx):
        path = self.samples[idx]
        with open(path, "r") as f:
            sample = json.load(f)

        x = torch.tensor(sample["node_features"], dtype=torch.float)
        edge_index = torch.tensor(sample["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(sample["edge_features"], dtype=torch.float)
        edge_mask = torch.tensor(sample["edge_mask"], dtype=torch.bool)
        y = torch.tensor(sample["target_edge"], dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_mask=edge_mask,
            y=y
        )

        return data

