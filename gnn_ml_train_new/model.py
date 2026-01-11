# model.py

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class EdgeMessagePassing(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr="add")  # sum messages
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, node_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_dim]

        row, col = edge_index
        src = x[row]
        dst = x[col]

        edge_input = torch.cat([src, dst, edge_attr], dim=1)
        edge_emb = self.edge_mlp(edge_input)

        return edge_emb


class SDRGNet(nn.Module):
    def __init__(self, node_dim=1, edge_dim=2, hidden_dim=64):
        super().__init__()

        self.edge_gnn = EdgeMessagePassing(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim
        )

        # Pointer / ranking head
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        """
        data.x          : [num_nodes, node_dim]
        data.edge_index : [2, num_edges]
        data.edge_attr  : [num_edges, edge_dim]
        data.edge_mask  : [num_edges]
        """

        edge_emb = self.edge_gnn(
            data.x,
            data.edge_index,
            data.edge_attr
        )

        scores = self.score_head(edge_emb).squeeze(-1)

        # Mask invalid edges
        if hasattr(data, "edge_mask"):
            scores = scores.masked_fill(~data.edge_mask, -1e9)

        return scores



