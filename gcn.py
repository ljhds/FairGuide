import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv




class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat, nhid, dropout)
        self.fc = nn.Linear(nhid, nclass)

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        x = self.fc(x)
        return x


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):

        x = F.relu(self.gc1(x,edge_index))
        x = self.dropout(x)
        x = self.gc2(x,edge_index)
        return x    