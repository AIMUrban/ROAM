import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GATv2Conv


class GNNLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_feat_dim=0, dropout=0.1):
        super().__init__(aggr='add')  # 'add', 'mean', or 'max'
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_feat_dim = edge_feat_dim

        self.node_lin1 = nn.Linear(in_dim, out_dim, bias=True)
        self.norm1 = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.edge_lin = nn.Linear(edge_feat_dim, in_dim) if edge_feat_dim > 0 else None

    def message(self, x_j, edge_attr=None, edge_weight=None):
        msg = x_j
        if edge_attr is not None:
            msg = msg + edge_attr
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)
        return msg

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None):
        if self.edge_feat_dim > 0:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)  # [E] → [E, 1]
            edge_attr_proj = self.edge_lin(edge_attr)  # [E, in_dim]
        else:
            edge_attr_proj = None

        org_x = x
        x = self.node_lin1(org_x)
        h = self.propagate(edge_index, x=x, edge_attr=edge_attr_proj, edge_weight=edge_weight)  # [N, in_dim]
        h = h + org_x
        h = self.norm1(h)
        h = self.act(h)
        h = self.drop(h)

        return h


class SpecificGNNEncoder(MessagePassing):
    def __init__(self, in_dim, out_dim, num_pools, edge_feat_dim=0, dropout=0.1):
        super().__init__(aggr='add')  # 'add', 'mean', or 'max'
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_pools = num_pools
        self.edge_feat_dim = edge_feat_dim

        if num_pools > 1:
            self.w_pool = nn.Parameter(torch.randn(num_pools, in_dim, out_dim))  # [P, in, out]
            self.bias_pool = nn.Parameter(torch.randn(num_pools, out_dim))  # [P, out]

        self.norm1 = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.node_lin = nn.Linear(in_dim, out_dim)

        self.edge_lin = nn.Linear(edge_feat_dim, in_dim) if edge_feat_dim > 0 else None

    def message(self, x_j, edge_weight=None, edge_attr=None):
        msg = x_j
        if edge_attr is not None:
            msg = msg + edge_attr
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)
        return msg

    def forward(self, x, proto_user_attn, edge_index=None, edge_weight=None, edge_attr=None):
        B, _ = proto_user_attn.shape

        if self.edge_feat_dim > 0:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)  # [E] → [E, 1]
            edge_attr_proj = self.edge_lin(edge_attr)  # [E, in_dim]
        else:
            edge_attr_proj = None

        agg_x = self.propagate(edge_index, x=x, edge_attr=edge_attr_proj)  # [N, in_dim]

        common_h = self.node_lin(agg_x)
        h = common_h
        h = self.norm1(h)
        h = self.act(h)
        h = self.drop(h)

        return h


class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, edge_feat_dim=0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()

        dims = [in_dim] + hidden_dims  # eg: [64, 128, 128]
        for i in range(len(dims) - 1):
            self.layers.append(
                GNNLayer(in_dim=dims[i], out_dim=dims[i+1], edge_feat_dim=edge_feat_dim, dropout=dropout),
            )

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None):
        for i, (layer) in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr, edge_weight)

        return x
