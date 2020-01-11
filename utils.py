import ast
import random
import time
import sys
import numpy as np
import pickle as pkl
import torch.nn.functional as F
from datetime import datetime
from functools import partial
import scipy.sparse as sp
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data


def load_candidates(filepath):
    myfile = open(filepath)
    lines = myfile.readlines()
    candidates = list()
    for line in lines:
        candidates.append(ast.literal_eval(line))
    myfile.close()
    return candidates

def generate_candidate(search_space, layers, nclass=None):
    arch = list()
    for _ in range(layers):
        arch.append(random.choice(search_space['attention_type']))
        arch.append(random.choice(search_space['aggregator_type']))
        arch.append(random.choice(search_space['activate_function']))
        arch.append(random.choice(search_space['number_of_heads']))
        arch.append(random.choice(search_space['hidden_units']))
    if nclass is not None:
        arch[-1] = nclass
    return arch

def get_arch_key(architecture):
    return '[' + ','.join(str(e) for e in architecture) + ']'

def save_history(history):
    time_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    history_file = open('log/history_{}'.format(time_str))
    for i, individul in enumerate(history):
        history_file.write('{}\t{}\t{}'.format(i, get_arch_key(individul.arch), individul.accuracy))
    history_file.close()

def eval_model(model, data, key, binary=False):
    model.eval()
    if not binary:
        act = partial(F.log_softmax, dim=1)
        criterion = F.nll_loss
    else:
        act = torch.sigmoid
        criterion = F.binary_cross_entropy

    with torch.no_grad():
        output = model(data.x, data.edge_index, data.edge_attr).squeeze()
        loss = criterion(act(output)[data.index_dict[key]], data.y[data.index_dict[key]])
        if not binary:
            predict_class = output.max(1)[1]
        else:
            predict_class = act(output).gt(0.5).float()
        correct = torch.eq(predict_class[data.index_dict[key]], data.y[data.index_dict[key]]).long().sum().item()
        acc = correct/len(data.index_dict[key])

    return {'loss': loss.item(), 'accuracy': acc}

def train_architecture(candidate, data, epochs=1000, early_stop=50, device='cuda'):
    start = time.perf_counter()
    data = data.to(device)
    model = candidate.build_gnn(data.num_features, data.y.max().item()+1)
    model = model.to(device)
    act = partial(F.log_softmax, dim=1)
    criterion = F.nll_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-6)
    val_losses = list()
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr).squeeze()
        loss = criterion(act(output)[data.index_dict['train']], data.y[data.index_dict['train']])
        loss.backward()
        optimizer.step()
        val_res = eval_model(model, data, 'val')
        val_losses.append(val_res["loss"])
        if early_stop and epoch > early_stop and val_losses[-1] > np.mean(val_losses[-early_stop+1:-1]):
            break
    train_time = time.perf_counter()-start
    return val_res['accuracy'], model, train_time

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).transpose().tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized
    
def sparse_to_torch_sparse(sparse_mx, device='cuda'):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    if device == 'cuda':
        indices = indices.cuda()
        values = torch.from_numpy(sparse_mx.data).cuda()
        shape = torch.Size(sparse_mx.shape)
        adj = torch.cuda.sparse.FloatTensor(indices, values, shape)
    elif device == 'cpu':
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        adj = torch.sparse.FloatTensor(indices, values, shape)
    return adj

def sparse_to_torch_dense(sparse, device='cuda'):
    dense = sparse.todense().astype(np.float32)
    torch_dense = torch.from_numpy(dense).to(device=device)
    return torch_dense

def load_corpus(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    index_dict = {}
    label_dict = {}
    phases = ["train", "val", "test"]
    objects = []
    def load_pkl(path):
        with open(path.format(dataset_str, p), 'rb') as f:
            if sys.version_info > (3, 0):
                return pkl.load(f, encoding='latin1')
            else:
                return pkl.load(f)

    for p in phases:
        index_dict[p] = load_pkl("data/ind.{}.{}.x".format(dataset_str, p))
        label_dict[p] = load_pkl("data/ind.{}.{}.y".format(dataset_str, p))

    adj = load_pkl("data/ind.{}.BCD.adj".format(dataset_str))
    adj = adj.astype(np.float32)
    adj = preprocess_adj(adj)

    label = np.zeros(adj.shape[0])
    for p in phases:
        np.put(label, index_dict[p], label_dict[p])

    return adj, index_dict, label

def get_data(dataset, device):
    sp_adj, index_dict, label = load_corpus(dataset)
    if dataset == "mr":
        label = torch.Tensor(label).to(device)
    else:
        label = torch.LongTensor(label).to(device)

    # features = torch.arange(sp_adj.shape[0]).to(device)
    # adj = sparse_to_torch_sparse(sp_adj, device=device)

    if dataset == "mr": nclass = 1
    else: nclass = label.max().item()+1
    adj_dense = sparse_to_torch_dense(sp_adj, device='cpu')

    edge_index, edge_attr = dense_to_sparse(adj_dense)
    data = Data(x=adj_dense.contiguous(), edge_index=edge_index, edge_attr=edge_attr, y=label)
    data.index_dict = index_dict
    return data, adj_dense.size()[1], nclass

def mutate_arch(individul, p=0.1, state_num=5):
    arch = individul.arch.copy()
    search_space = individul.search_space
    assert len(arch) % state_num == 0
    layers = len(arch) // state_num
    for i in range(layers):
        for k in range(state_num):
            if random.random() < p:
                if k == 0:
                    key = "attention_type"
                elif k == 1:
                    key = "aggregator_type"
                elif k == 2:
                    key = "activate_function"
                elif k == 3:
                    key = "number_of_heads"
                elif k == 4:
                    key = "hidden_units"
                arch[i*state_num + k] = random.choice(search_space[key])
    return arch
        