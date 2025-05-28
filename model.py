import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

seed=20
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class MLP_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    

class APPNP(torch.nn.Module):
    def __init__(self, K=5, alpha=0.1):
        super(APPNP, self).__init__()
        
        self.K = K  
        self.alpha = alpha 

        

    def forward(self, x, adj):

        deg_inv_sqrt = adj.sum(dim=1).pow(-0.5)


        # Avoid calculation exceptions
        deg_inv_sqrt = torch.where(deg_inv_sqrt == float('inf'), torch.tensor(1e-8, device=deg_inv_sqrt.device), deg_inv_sqrt)
        

        adj_norm = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

        
        h = x  
        for _ in range(self.K):
            x = torch.matmul(adj_norm, x)
            x = (1 - self.alpha) * x + self.alpha * h  # restart
        x=torch.softmax(x,dim=1)
        return x


class PseudoCommunityModel(nn.Module):
    def __init__(self, sens, alpha=0.1, K=5):
        super(PseudoCommunityModel, self).__init__()
        self.appnp = APPNP(K=K, alpha=alpha)
        self.sens=sens

    def forward(self, one_hot_labels, adj):

        propagated_labels = self.appnp(one_hot_labels, adj)
        
        return propagated_labels
    