from torch_geometric.nn import GATConv
import torch.nn as nn
import torch
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, nfeat, nclass, hidden=64, heads=1, dropout=0):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(nfeat, hidden, heads=heads))
        self.convs.append(GATConv(heads*hidden, hidden, heads=heads))
        self.convs.append(GATConv(heads*hidden, nclass, heads=heads, concat=False))
    
    def forward(self, x, edge_index, edge_weight=None):
        out = F.leaky_relu(self.convs[0](x, edge_index))
        out = F.dropout(out, self.dropout, self.training)
        out = F.leaky_relu(self.convs[1](out, edge_index))
        out = F.dropout(out, self.dropout, self.training)
        out = self.convs[2](out, edge_index)
        return out