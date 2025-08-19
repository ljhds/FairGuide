import time
import argparse
import numpy as np

import torch
print(torch.__version__)

import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
import scipy.sparse as sp
from utils import load_github,load_pokec_n,load_pokec_z
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from torch_geometric.utils import from_scipy_sparse_matrix
import torch_geometric.transforms as T
from torch_geometric.data import Data
from tqdm import tqdm


# ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

torch.cuda.set_device(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



#Working dictionary
import os
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
print("Current working directory:", os.getcwd())

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--cuda_device', type=int, default=0,
                    help='cuda device running on.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--dataset', type=str, default='github',
                    help='a dataset from credit, german and bail.')
parser.add_argument('--FG', type=int, default=1,
                    help='1 and 0 represent utilizing and not utilizing the FairGuide data.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(args.cuda_device)
device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
seed=50

import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if args.dataset =='github':
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_github('github')
elif args.dataset =='pokec_z':
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec_z('pokec_z')
elif args.dataset =='pokec_n':
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec_n('pokec_n')


if args.FG:
    adj= sp.load_npz(f'debiased_data/{args.dataset}.npz')

def calculate_delta_sp(predicted_labels, sens, num_communities):

    delta_sp = 0
    
    # node of s=0 and s=1
    nodes_s0 = np.where(sens == 0)[0]
    nodes_s1 = np.where(sens == 1)[0]
    
    for c in range(num_communities):
        # node belong to community c
        community_nodes = np.where(predicted_labels == c)[0]

        # node of s = 0 and s = 1 belong to community c
        community_s0 = np.intersect1d(community_nodes, nodes_s0)  # 社区c且s=0的节点
        community_s1 = np.intersect1d(community_nodes, nodes_s1)  # 社区c且s=1的节点
        
        p_s0 = len(community_s0) / len(nodes_s0) if len(nodes_s0) > 0 else 0
        p_s1 = len(community_s1) / len(nodes_s1) if len(nodes_s1) > 0 else 0
        
        delta_sp += np.abs(p_s0 - p_s1)

    delta_sp /= 2

    
    return delta_sp


import numpy as np
import networkx as nx
import community as community_louvain  # python-louvain


G = nx.from_scipy_sparse_array(adj)

# Run Louvain community detection
partition = community_louvain.best_partition(G)  

labels_pred = np.array([partition[node] for node in sorted(G.nodes())])

sens = np.array(sens)

num_communities = len(set(labels_pred))
delta_sp = calculate_delta_sp(labels_pred, sens, num_communities)

print(f"Delta_SP: {delta_sp:.4f}")