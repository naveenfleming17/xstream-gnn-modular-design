import torch
import torch.nn as nn
from dgl.nn import GraphConv, SumPooling, EdgeWeightNorm

class ModifiedGCN(nn.Module):
    def __init__(self, input_dim, edge_dim, output_dim):
        super().__init__()

        hidden_dims = [32, 32, 32, 16]

        self.conv1 = GraphConv(input_dim, hidden_dims[0], allow_zero_in_degree=True)
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])

        self.conv2 = GraphConv(hidden_dims[0], hidden_dims[1], allow_zero_in_degree=True)
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])

        self.conv3 = GraphConv(hidden_dims[1], hidden_dims[2], allow_zero_in_degree=True)
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])

        self.conv4 = GraphConv(hidden_dims[2], hidden_dims[3], allow_zero_in_degree=True)
        self.bn4 = nn.BatchNorm1d(hidden_dims[3])

        self.pool = SumPooling()
        self.classify = nn.Linear(hidden_dims[3] + edge_dim, output_dim)

        self.elu = nn.ELU()
        self.edge_weight_norm = EdgeWeightNorm()

    def forward(self, g, node_features, edge_features, edge_weights):
        edge_weights = torch.abs(edge_weights)
        edge_weights = self.edge_weight_norm(g, edge_weights)

        x = self.elu(self.bn1(self.conv1(g, node_features, edge_weight=edge_weights)))
        x = self.elu(self.bn2(self.conv2(g, x, edge_weight=edge_weights)))
        x = self.elu(self.bn3(self.conv3(g, x, edge_weight=edge_weights)))
        x = self.elu(self.bn4(self.conv4(g, x, edge_weight=edge_weights)))

        x = self.pool(g, x)

        edge_mean = torch.mean(edge_features, dim=0).unsqueeze(0)
        edge_mean = edge_mean.repeat(x.shape[0], 1)

        x = torch.cat([x, edge_mean], dim=1)

        return self.classify(x)
