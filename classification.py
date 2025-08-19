import argparse
import numpy as np

import torch
print(torch.__version__)

import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import  roc_auc_score, f1_score
import scipy.sparse as sp


from utils import load_github,load_pokec_z,load_pokec_n
from gcn import GCN
from torch_geometric.utils import convert
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')




# ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')


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
parser.add_argument('--dataset', type=str, default='pokec_n',
                    help='github or pokec.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parsameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
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

labels[labels>1]=1
sens[sens>0]=1

edge_index = convert.from_scipy_sparse_matrix(adj)[0].cuda()


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()

if args.cuda:
    torch.cuda.set_device(args.cuda_device)
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

def train_and_evaluate():

    def train(epoch, pa, eq, test_f1, val_loss, test_auc):
        model.train()
        optimizer.zero_grad()
        
        output = model(x = features, edge_index=torch.LongTensor(edge_index.cpu()).cuda())

        preds = (output.squeeze() > 0).type_as(labels)

        loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())

        auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_train.cpu().numpy()], output.detach().cpu().numpy()[idx_train.cpu().numpy()])
        f1_train = f1_score(labels[idx_train.cpu().numpy()].cpu().numpy(), preds[idx_train.cpu().numpy()].cpu().numpy())
        loss_train.backward()
        optimizer.step()
        _, _ = fair_metric(preds[idx_train.cpu().numpy()].cpu().numpy(), labels[idx_train.cpu().numpy()].cpu().numpy(), sens[idx_train.cpu().numpy()].cpu().numpy())

        model.eval()
        output = model(x= features, edge_index=torch.LongTensor(edge_index.cpu()).cuda())
        preds = (output.squeeze() > 0).type_as(labels)

        loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float())
        auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val.cpu().numpy()], output.detach().cpu().numpy()[idx_val.cpu().numpy()])
        f1_val = f1_score(labels[idx_val.cpu().numpy()].cpu().numpy(), preds[idx_val.cpu().numpy()].cpu().numpy())

        if epoch < 15:
            return 0, 0, 0, 1e5, 0 
        if loss_val < val_loss:
            val_loss = loss_val.data
            pa, eq, test_f1, test_auc = test(test_f1)
            
        return pa, eq, test_f1, val_loss, test_auc

    def test(test_f1):
        model.eval()
        output = model(x = features, edge_index=torch.LongTensor(edge_index.cpu()).cuda())
        preds = (output.squeeze() > 0).type_as(labels)


        loss_test = F.binary_cross_entropy_with_logits(output[idx_test], labels[idx_test].unsqueeze(1).float())
        auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu().numpy()], output.detach().cpu().numpy()[idx_test.cpu().numpy()])
        f1_test = f1_score(labels[idx_test.cpu().numpy()].cpu().numpy(), preds[idx_test.cpu().numpy()].cpu().numpy())

        test_auc = auc_roc_test
        test_f1 = f1_test

        parity_test, equality_test = fair_metric(preds[idx_test.cpu().numpy()].cpu().numpy(),
                                                labels[idx_test.cpu().numpy()].cpu().numpy(),
                                                sens[idx_test.cpu().numpy()].cpu().numpy())
        


        return parity_test, equality_test, test_f1, test_auc


    model = GCN(nfeat = features.shape[1], nhid=args.hidden, nclass=labels.max().item(), dropout=args.dropout).float()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.cuda:
        torch.cuda.set_device(args.cuda_device)
        model.cuda()
    # Train model
    val_loss = 1e5
    pa = 0
    eq = 0
    test_auc = 0
    test_f1 = 0
    for epoch in tqdm(range(args.epochs)):
        pa, eq, test_f1, val_loss, test_auc = train(epoch, pa, eq, test_f1, val_loss, test_auc)
    
    print("Delta_{SP}: " + str(pa))
    print("Delta_{EO}: " + str(eq))
    print("F1: " + str(test_f1))
    print("AUC: " + str(test_auc))

train_and_evaluate()
