import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FeaStConv, GATConv, GCNConv

class MeshGNN(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64, out_channels=1):
        super().__init__()
        self.conv1 = FeaStConv(in_channels, hidden_channels)
        self.conv2 = FeaStConv(hidden_channels, hidden_channels)
        self.conv3 = FeaStConv(hidden_channels, out_channels)

        # Projections pour résidus si dimensions différentes
        self.res1 = nn.Linear(in_channels, hidden_channels)
        self.res2 = nn.Identity()  # hidden -> hidden (même taille)
        self.res3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, pos = data.x, data.edge_index, data.x[:, :3]  # pos = x, y, z only

        # Layer 1
        x1 = F.relu(self.conv1(x, edge_index))
        x = x1 + self.res1(x)

        # Layer 2
        x2 = F.relu(self.conv2(x, edge_index))
        x = x2 + self.res2(x)

        # Layer 3
        x3 = self.conv3(x, edge_index)
        out = x3 + self.res3(x)  # pas de ReLU à la fin (régression)

        return out

class MeshGNNGAT(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64, out_channels=1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, out_channels)

        # Projections pour résidus si dimensions différentes
        self.res1 = nn.Linear(in_channels, hidden_channels)
        self.res2 = nn.Identity()  # hidden -> hidden (même taille)
        self.res3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, pos = data.x, data.edge_index, data.x[:, :3]  # pos = x, y, z only

        # Layer 1
        x1 = F.relu(self.conv1(x, edge_index))
        x = x1 + self.res1(x)

        # Layer 2
        x2 = F.relu(self.conv2(x, edge_index))
        x = x2 + self.res2(x)

        # Layer 3
        x3 = self.conv3(x, edge_index)
        out = x3 + self.res3(x)  # pas de ReLU à la fin (régression)

        return out
    
class MeshGNNGCNConv(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64, out_channels=1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

        # Projections pour résidus si dimensions différentes
        self.res1 = nn.Linear(in_channels, hidden_channels)
        self.res2 = nn.Identity()  # hidden -> hidden (même taille)
        self.res3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, pos = data.x, data.edge_index, data.x[:, :3]  # pos = x, y, z only

        # Layer 1
        x1 = F.relu(self.conv1(x, edge_index))
        x = x1 + self.res1(x)

        # Layer 2
        x2 = F.relu(self.conv2(x, edge_index))
        x = x2 + self.res2(x)

        # Layer 3
        x3 = self.conv3(x, edge_index)
        out = x3 + self.res3(x)  # pas de ReLU à la fin (régression)

        return out