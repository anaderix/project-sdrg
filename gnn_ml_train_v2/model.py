# model.py
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class EdgeConditionedMessagePassing(MessagePassing):
    """
    Real message-passing layer.

    For each directed edge j -> i, the message depends on:
        h_i, h_j, e_ij

    Node embeddings are updated by aggregating messages from neighbors.
    """

    def __init__(self, hidden_dim, edge_dim, dropout=0.0):
        super().__init__(aggr="add")

        self.message_mlp = MLP(
            in_dim=2 * hidden_dim + edge_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )

        self.update_mlp = MLP(
            in_dim=2 * hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_attr):
        m = self.propagate(
            edge_index=edge_index,
            h=h,
            edge_attr=edge_attr,
        )

        h_new = self.update_mlp(torch.cat([h, m], dim=-1))

        # residual update
        h = self.norm(h + h_new)

        return h

    def message(self, h_i, h_j, edge_attr):
        msg_input = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)


class SDRGNet(nn.Module):
    """
    Message-passing GNN for SDRG bond selection.

    Steps:
        1. Encode node features.
        2. Encode edge features.
        3. Perform L rounds of edge-conditioned message passing.
        4. Score every active bond using updated node embeddings and edge embeddings.
    """

    def __init__(
        self,
        node_dim=1,
        edge_dim=3,
        hidden_dim=64,
        num_layers=3,
        dropout=0.0,
    ):
        super().__init__()

        self.node_encoder = MLP(
            in_dim=node_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )

        self.edge_encoder = MLP(
            in_dim=edge_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )

        self.layers = nn.ModuleList([
            EdgeConditionedMessagePassing(
                hidden_dim=hidden_dim,
                edge_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.score_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        for layer in self.layers:
            h = layer(h, edge_index, e)

        row, col = edge_index
        h_i = h[row]
        h_j = h[col]

        directed_score_input = torch.cat([h_i, h_j, e], dim=-1)
        directed_scores = self.score_mlp(directed_score_input).squeeze(-1)

        # Keep only canonical i -> j physical edges.
        # edge_mask = True for one direction of each physical bond.
        if hasattr(data, "edge_mask"):
            scores = directed_scores[data.edge_mask]
        else:
            scores = directed_scores

        return scores