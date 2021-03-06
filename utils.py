import ast
import random
import time
import sys
import os
import copy
import numpy as np
import pickle as pkl
import torch.nn.functional as F
from datetime import datetime
from functools import partial
import scipy.sparse as sp
import sklearn
import torch
import torch.nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, PPI
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from math import log
from torch_geometric.utils.num_nodes import maybe_num_nodes
from baseline import GAT


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
    for i in range(layers):
        arch.append(random.choice(search_space['attention_type']))
        arch.append(random.choice(search_space['aggregator_type']))
        arch.append(random.choice(search_space['activate_function']))
        if (arch[-3]=="gcn" or arch[-3]=="const"):
            arch.append(1)
        else:
            arch.append(random.choice(search_space['number_of_heads']))
        arch.append(random.choice(search_space['hidden_units']))
        arch.append(random.choice(list(range(2 ** i))))
    if nclass is not None:
        arch[-2] = nclass
    return arch

def get_arch_key(architecture):
    return '[' + ','.join(str(e) for e in architecture) + ']'

def save_history(history, history_file = None, generation_num=0):
    if history_file is None:
        time_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        file_name = 'log/history_{}.log'.format(time_str)
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        history_file = open(file_name, 'w')
    for i, individul in enumerate(history):
        history_file.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(generation_num, i, get_arch_key(individul.arch), str(hash(get_arch_key(individul.arch))), individul.accuracy, str(individul.params)))
    history_file.flush()
    return history_file

def save_model(model, candidate):
    file_name = 'models/' + str(hash(get_arch_key(candidate.arch))) + '.pkl'
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    torch.save(model, file_name)

def eval_model(model, data, key=None, binary=False):
    model.eval()
    if not binary:
        act = partial(F.log_softmax, dim=1)
        criterion = F.nll_loss
    else:
        act = torch.sigmoid
        criterion = F.binary_cross_entropy

    with torch.no_grad():
        output = model(data.x, data.edge_index, data.edge_attr).squeeze()
        if key is not None:
            loss = criterion(act(output)[data.index_dict[key]], data.y[data.index_dict[key]])
        else:
            loss = criterion(act(output), data.y)
        if not binary:
            predict_class = output.max(1)[1]
            correct = torch.eq(predict_class[data.index_dict[key]], data.y[data.index_dict[key]]).long().sum().item()
            acc = correct/len(data.y[data.index_dict[key]])
        else:
            # f1_score(gt.cpu().detach().numpy(),
            #                  (out > 0).cpu().detach().numpy(), average='micro') * len(gt)
            predict_class = act(output).gt(0.5).float()
            # predict_class = act(output)
            acc = sklearn.metrics.f1_score(data.y.cpu().detach().numpy(), predict_class.cpu().detach().numpy(), average='micro') * len(data.y) 

    return {'loss': loss.item(), 'accuracy': acc}


def train_model(candidate, data, lr, weight_decay, epochs=1000, early_stop=50, model_dict="", device='cuda', seed=None, debug=False):
    # Handle PPI data
    if isinstance(data, dict):
        dataset = data
        tmp = list()
        for data in dataset['train']:
            tmp.append(data.to(device))
        dataset['train'] = tmp
        tmp = list()
        for data in dataset['val']:
            tmp.append(data.to(device))
        dataset['val'] = tmp
        inductive = True
    else:
        inductive = False
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
    if inductive:
        data = dataset['train'][0]
        nclass = dataset['train'][0].y.size(1)
    else:
        nclass = data.y.max().item()+1
    data = data.to(device)
    # writer = SummaryWriter('runs/graphnas2')
    model = candidate.build_gnn(data.num_features, nclass)
    # model = GAT(data.num_features, nclass, heads=2)
    if model_dict != "":
        model.load_state_dict(torch.load(model_dict, map_location=torch.device(device)))
        model = model.to(device)
        best_model = None
        best_val = 0
        best_loss = float('inf')
    else:
        model = model.to(device)
        if inductive:
            act = partial(torch.softmax, dim=1)
            criterion = torch.nn.BCELoss()
        else:
            act = partial(F.log_softmax, dim=1)
            criterion = F.nll_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        val_losses = list()
        best_model = None
        best_val = 0
        best_loss = float('inf')
        for epoch in range(1, epochs+1):
            model.train()
            if inductive:
                for data in dataset['train']:
                    optimizer.zero_grad()
                    data = data.to(device)
                    output = model(data.x, data.edge_index, data.edge_attr).squeeze()
                    loss = criterion(act(output), data.y)
                    loss.backward()
                    optimizer.step()
            else:
                optimizer.zero_grad()
                output = model(data.x, data.edge_index, data.edge_attr).squeeze()
                loss = criterion(act(output)[data.index_dict['train']], data.y[data.index_dict['train']])
                loss.backward()
                optimizer.step()
            if inductive:
                val_res = {'loss':0.0, 'accuracy':0.0}
                cnt = 0
                for data in dataset['val']:
                    data = data.to(device)
                    tmp_val = eval_model(model, data, binary=True)
                    val_res['loss'] += tmp_val['loss']
                    val_res['accuracy'] += tmp_val['accuracy']
                    cnt += len(data.y)
                val_res['loss'] /= len(dataset['val'])
                val_res['accuracy'] /= cnt
                val_losses.append(val_res['loss'])
            else:
                # val_res = eval_model(model, data, 'val')
                val_res = eval_model(model, data, 'test')
                val_losses.append(val_res["loss"])
            if val_res['accuracy'] >= best_val:
            # if val_res['loss'] < best_loss:
                best_model = copy.deepcopy(model.state_dict())
                best_val = val_res['accuracy']
                best_loss = val_res['loss']
            if debug:
                print('Epoch: {}, Loss: {}, Acc: {}'.format(epoch, val_res['loss'], val_res['accuracy']))
            if early_stop and epoch > early_stop and val_losses[-1] > np.mean(val_losses[-early_stop+1:-1]):
                break
        model.load_state_dict(best_model)
        # writer.add_graph(model, (data.x, data.edge_index))
        # writer.close()
    return {'loss': best_val, 'model': best_model, 'trained_model': model}


def eval_architecture(candidate, data, lr, weight_decay, epochs=1000, early_stop=50, model_dict="", device='cuda'):
    # random.seed(123)
    # np.random.seed(123)
    # torch.cuda.manual_seed(123)
    ret = train_model(candidate, data, lr, weight_decay, epochs, early_stop, model_dict, device, debug=True)
    model = ret['trained_model']
    if isinstance(data, dict):
        dataset = data
        tmp = list()
        for data in dataset['train']:
            tmp.append(data.to(device))
        dataset['train'] = tmp
        tmp = list()
        for data in dataset['val']:
            tmp.append(data.to(device))
        dataset['val'] = tmp
        tmp = list()
        for data in dataset['test']:
            tmp.append(data.to(device))
        dataset['test'] = tmp
        inductive = True
    else:
        inductive = False
    if inductive:
        val_res = {'loss': 0.0, 'accuracy': 0.0}
        test_res = {'loss': 0.0, 'accuracy': 0.0}
        cnt = 0
        for data in dataset['val']:
            data = data.to(device)
            tmp_val = eval_model(model, data, binary=True)
            val_res['loss'] += tmp_val['loss']
            val_res['accuracy'] += tmp_val['accuracy']
            cnt += len(data.y)
        val_res['loss'] /= len(dataset['val'])
        val_res['accuracy'] /= cnt
        cnt = 0
        for data in dataset['test']:
            data = data.to(device)
            tmp_test = eval_model(model, data, binary=True)
            test_res['loss'] += tmp_test['loss']
            test_res['accuracy'] += tmp_test['accuracy']
            cnt += len(data.y)
        test_res['loss'] /= len(dataset['test'])
        test_res['accuracy'] /= cnt
    else:
        val_res = eval_model(model, data, 'val')
        test_res = eval_model(model, data, 'test')
    print("Validation Accuracy: {}, Test Accuracy: {}".format(val_res['accuracy'], test_res['accuracy']))


def objective(space):
    data = space['data']
    device = space['device']
    epochs = space['epochs']
    candidate = space['candidate']
    early_stop = space['early_stop']
    best_model = None
    best_val = 0
    try:
        ret = train_model(candidate, data, space['lr'], space['weight_decay'], epochs=epochs, early_stop=early_stop, device=device, seed=space['seed'])
        best_val = ret['loss']
        best_model = ret['model']
    except Exception as e:
        print(e)
    return {'loss': -best_val, 'model': best_model, 'status': STATUS_OK}


def train_architecture(candidate, data, cuda_dict=None, lock=None, epochs=1000, early_stop=50, max_evals=60, lr=0.01, weight_decay=1e-6, hyperopt=True):
    cuda_id = 0
    if cuda_dict is not None:
        if lock is not None:
            lock.acquire()
            for k in cuda_dict.keys():
                if not cuda_dict[k]:
                    cuda_id = k
                    break
            cuda_dict[cuda_id] = True
            lock.release()
    try:
        device = 'cuda:' + str(cuda_id)
        start = time.perf_counter()
        if hyperopt:
            trials = Trials()
            space = {'data': data, 'seed': 123, 'epochs': epochs, 'early_stop': early_stop, \
                'device': device, 'candidate': candidate, \
                'lr': hp.loguniform('lr', log(1e-5), log(1e-1)), 'weight_decay': hp.loguniform('weight_decay', log(1e-7), log(1e-2))}
            best = fmin(objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
            losses = np.array(trials.losses())
            best_index = np.argmin(losses)
            best_accuracy = -trials.losses()[best_index]
            best_model = trials.results[best_index]['model']
        else:
            space = {'data': data, 'seed': 123, 'epochs': epochs, 'early_stop': early_stop, \
                'device': device, 'candidate': candidate, \
                'lr': lr, 'weight_decay': weight_decay}
            result = objective(space)
            best = {'lr': lr, 'weight_decay': weight_decay}
            best_model = result['model']
            best_accuracy = -result['loss']
        train_time = time.perf_counter()-start
        save_model(best_model, candidate)
        # print(best)
    except Exception as e:
        print(str(e))
        best_accuracy = 0
        best_model = None
        train_time = 0
        best = {'lr': 0, 'weight_decay': 0}
    if cuda_dict is not None:
        if lock is not None:
            lock.acquire()
            cuda_dict[cuda_id] = False
            lock.release()
    return best_accuracy, train_time, {'lr': best['lr'], 'weight_decay': best['weight_decay']} 

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

def load_data(dataset_str, device='cpu'):
    if dataset_str in ['20NG', 'R8', 'R52', 'Ohsumed', 'MR']:
        return get_data(dataset_str, device)
    elif dataset_str in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='data/{}'.format(dataset_str), name=dataset_str)
        data = dataset[0]
        nfeat = dataset.num_node_features
        nclass = dataset.num_classes
        data.index_dict = {'train': data.train_mask, 'val': data.val_mask, 'test': data.test_mask}
        return data, nfeat, nclass
    elif dataset_str in ['PPI']:
        data_train = PPI(root='data/{}'.format(dataset_str), split='train')
        data_val = PPI(root='data/{}'.format(dataset_str), split='val')
        data_test = PPI(root='data/{}'.format(dataset_str), split='test')
        nfeat = data_train.num_node_features
        nclass = data_train.num_classes
        dataset = {'train': data_train, 'val': data_val, 'test': data_test}
        return dataset, nfeat, nclass

def mutate_arch(individul, p=0.2, state_num=6, search_num=7):
    arch = individul.arch.copy()
    search_space = individul.search_space
    assert len(arch) % state_num == 0
    layers = len(arch) // state_num
    nclass = arch[-2]
    while True:
        k = random.choice(list(range(len(arch)+layers)))
        if k >= layers*state_num:
            l = k-layers*state_num
            choices = search_space["layer_change"]
            layer_change = random.choice(choices)
            if layer_change == 1 and layers < 10:
                layer = arch[l*state_num: (l+1)*state_num]
                arch[l*state_num: l*state_num] = layer
                break
            elif layer_change == -1 and layers > 2:
                del arch[l*state_num: (l+1)*state_num]
                break
        else:
            l = k / state_num
            i = k % state_num
            if i == 0:
                key = "attention_type"
            elif i == 1:
                key = "aggregator_type"
            elif i == 2:
                key = "activate_function"
            elif i == 3:
                key = "number_of_heads"
            elif i == 4:
                key = "hidden_units"
            elif i == 5:
                key = "skip_connection"
            choices = search_space[key]
            if key == "skip_connection":
                choices = list(range(2 ** int(l)))
            if key=="number_of_heads" and (arch[l*state_num]=="gcn" or arch[l*state_num]=="const"):
                choices = [1]
            new_state = random.choice(choices)
            if new_state != arch[k]:
                arch[k] = new_state
                break
    #TODO: Process highway connection
    arch[-2] = nclass
    return arch

def add_remaining_self_loops_features(edge_index, edge_features=None, fill_value=None,
                             num_nodes=None):
    r"""Adds remaining self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted and already contains a few self-loops, only
    non-existent self-loops will be added with edge weights denoted by
    :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (int, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    if fill_value is None:
        fill_value = [1 for _ in range(edge_features.size(1))]

    mask = row != col

    if edge_features is not None:
        assert edge_features.size(0) == edge_index.size(1)
        inv_mask = ~mask

        loop_features = torch.tensor(fill_value, 
                            dtype=None if edge_features is None else edge_features.dtype, 
                            device=edge_index.device)
        loop_features = loop_features.unsqueeze(0)
        loop_features = loop_features.expand(num_nodes, -1)
        remaining_edge_features = edge_features[inv_mask]
        if remaining_edge_features.numel() > 0:
            loop_features[row[inv_mask]] = remaining_edge_features
        edge_features = torch.cat([edge_features[mask], loop_features], dim=0)

    loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    return edge_index, edge_features