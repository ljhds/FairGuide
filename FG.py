
import time
import argparse
import numpy as np

import torch
import torch.nn as nn

print(torch.__version__)

import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
from utils import load_pokec,load_github,fairness_loss_fn
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import ctypes
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.cluster import KMeans
from model import PseudoCommunityModel,MLP_encoder
import os


import os
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
print("Current working directory:", os.getcwd())

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--cuda_device', type=int, default=0,
                    help='cuda device running on.')
parser.add_argument('--dataset', type=str, default='github',
                    help='a dataset from pokec and github.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(args.cuda_device)
device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')

seed = 10
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

if args.dataset =='pokec_n':
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec('pokec_n')
elif args.dataset =='pokec_z':
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec('pokec_z')
elif args.dataset =='github':
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_github('github')



def gumbel_edge_sampling(grad, adj_matrix, sens, edge_batch = 100, temperature=1.0, block_size=15000, cross_group_boost = 3.0):

    N = grad.size(0)
    
    sens = sens.to(device)
    
    global_weights = torch.tensor([], dtype=torch.float16, device=device)
    global_indices = torch.tensor([], dtype=torch.long, device=device)  
    j=0
    j_end=N

    # Block processing
    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)

        sens_i = sens[i:i_end]  
        sens_j = sens[j:j_end]  
        
        # inter_sens_boost
        cross_mask = (sens_i.unsqueeze(1) != sens_j.unsqueeze(0)).to(torch.float16)
        
        grad_block = grad[i:i_end, j:j_end]
        adj_block = adj_matrix[i:i_end, j:j_end]
        
        weights_block = torch.clamp(-grad_block, min=0)
        
        
        weights_block = weights_block * (1 + (cross_group_boost - 1) * cross_mask).half()
        
        mask = (adj_block == 0).float().to(device).half()
        weights_block = weights_block * mask
        
        flat_weights = weights_block.flatten().to(device)
        rows = torch.arange(i, i_end, device=device).unsqueeze(1)
        cols = torch.arange(j, j_end, device=device).unsqueeze(0)
        #get global indices of the block
        indices_block = (rows * N + cols).flatten()
        
        valid_mask = (flat_weights > 0).to(device)

        global_weights = torch.cat([global_weights, flat_weights[valid_mask]])
        global_indices = torch.cat([global_indices, indices_block[valid_mask]])
        
    


    
    if len(global_weights) == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
    
    # Gumbel-max Sampling
    logits = torch.log(global_weights + 1e-8)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    noisy_logits = (logits + gumbel_noise) / temperature
    
    k = min(edge_batch, len(global_weights))
    _, topk_indices = torch.topk(noisy_logits, k)
    selected_indices = global_indices[topk_indices]
    
    # trans indices
    selected_rows = selected_indices // N
    selected_cols = selected_indices % N
    
    return selected_rows, selected_cols


def update_graph_structure(adj_matrix, one_hot_labels, sensitive, edge_batch = 100):
    """
    adj updating
    """

    adj_matrix = adj_matrix.detach().requires_grad_(True)

    propagated_labels = model(one_hot_labels, adj_matrix)


    fairness_loss = fairness_loss_fn(propagated_labels, sensitive)

    grad = torch.autograd.grad(fairness_loss, adj_matrix, retain_graph=False, create_graph=False)[0]

    adj_matrix = adj_matrix.detach()


    #New edge sampling
    selected_rows, selected_cols = gumbel_edge_sampling(
        grad, adj_matrix,sensitive, edge_batch = edge_batch
    )

    adj_matrix[selected_rows, selected_cols] = 1.0
    
    return adj_matrix


def train_autoencoder(autoencoder, features, epochs=100, lr=0.01):
    autoencoder.train()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss() 

    for epoch in range(epochs):
        optimizer.zero_grad()

        encoded, decoded = autoencoder(features)

        loss = criterion(decoded, features)  
        loss.backward()

        optimizer.step()

edge_index,edge_weight=from_scipy_sparse_matrix(adj)

mlp_encoder = MLP_encoder(input_dim=features.shape[1], hidden_dim=16, output_dim=8).to(device)


features = torch.FloatTensor(features).to(device)  

train_autoencoder(mlp_encoder, features, epochs=1000, lr=0.005)

with torch.no_grad():
    encoded_features, _ = mlp_encoder(features)

# model = PseudoCommunityModel(sens=sens,K=1).to(device) # pokec-z
# model = PseudoCommunityModel(sens=sens,K=6).to(device) # pokec-n
model = PseudoCommunityModel(sens=sens,K=5).to(device) 

adj = torch.tensor(adj.toarray(), dtype=torch.float16,requires_grad=False).to(device)
encoded_features=encoded_features.half()

kmeans = KMeans(n_clusters = 20, random_state=42)
initial_labels = kmeans.fit_predict(encoded_features.cpu().numpy())
initial_labels = torch.tensor(initial_labels, dtype=torch.long, device=encoded_features.device)
one_hot_labels = F.one_hot(initial_labels, num_classes = 20).float().to(adj.device).half()


adding_percent = 0.04
num_edges = torch.count_nonzero(adj).item()
epochs = int(adding_percent * num_edges / 100)
for epoch in tqdm(range(epochs)):
    adj = update_graph_structure(adj, one_hot_labels, sens)


adj_numpy = adj.cpu().numpy().astype('float32')
adj_sparse = sp.csr_matrix(adj_numpy)
sp.save_npz(f'{args.dataset}.npz', adj_sparse)






