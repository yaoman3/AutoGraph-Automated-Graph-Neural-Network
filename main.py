import argparse
import time
import multiprocessing
import collections
import random
import torch
from search_space import MacroSearchSpace
from model import Individual
import utils


def build_args():
    parser = argparse.ArgumentParser(description='NAS for Graph')
    register_default_args(parser)
    args = parser.parse_args()
    return args


def register_default_args(parser):
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='train: Search Neural Architectures, test: Evaluate Architectures')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    parser.add_argument('--evolution_size', type=int, default=2000)
    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--sample_size', type=int, default=10)
    parser.add_argument('--init_candidates', type=str, default="")
    parser.add_argument('--search_space', type=str, default="MacroSearchSpace")
    parser.add_argument('--search_layers', type=int, default=2)
    # child model
    parser.add_argument("--dataset", type=str, default="R8", required=False,
                        help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument("--multi_label", type=bool, default=False,
                        help="multi_label or single_label task")
    parser.add_argument("--residual", action="store_false",
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.5,
                        help="input feature dropout")
    parser.add_argument("--hyperopt", action="store_true",
                        help="do hyperparameter search")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--max_param', type=float, default=5E6)
    parser.add_argument('--test_structure', type=str, default="")


def main(args): 
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
    device = 'cuda' if args.cuda else 'cpu'

    if args.search_space == "MacroSearchSpace":
        search_space = MacroSearchSpace().get_search_space()

    data, nfeat, nclass = utils.get_data(args.dataset, device)

    population = collections.deque()
    if args.init_candidates:
        archs = utils.load_candidates(args.init_candidates)
        for arch in archs:
            candidate = Individual(search_space, nfeat=nfeat, nclass=nclass)
            candidate.arch = arch
            population.append(candidate)
    while len(population)<args.population_size:
        candidate = Individual(search_space, nfeat=nfeat, nclass=nclass)
        candidate.arch = utils.generate_candidate(search_space, args.search_layers, nclass)
        population.append(candidate)

    history = list()
    model_dict = dict()
    best_accuracy = 0.0
    for candidate in population:
        arch_str = utils.get_arch_key(candidate.arch)
        if arch_str not in model_dict:
            try:
                candidate.accuracy, candidate.trained_model, candidate.train_time = utils.train_architecture(candidate, data)
            except:
                candidate.accuracy = 0.0
            model_dict[arch_str] = candidate
            if candidate.accuracy > best_accuracy:
                best_accuracy = candidate.accuracy
            print("Architecture: {}\tAccuracy: {}\tCurrent Best: {}\n".format(arch_str, candidate.accuracy, best_accuracy))
        else:
            candidate.accuracy = model_dict[arch_str].accuracy
            candidate.trained_model = model_dict[arch_str].trained_model
            candidate.train_time = model_dict[arch_str].train_time
        history.append(candidate)

    while len(history)<args.evolution_size:
        sample = list()
        while len(sample)<args.sample_size:
            candidate = random.choices(list(population))
            sample.append(candidate)
        parent = max(sample, key=lambda i: i.accuracy)
        child = Individual(search_space, nfeat=nfeat, nclass=nclass)
        child.arch = utils.mutate_arch(parent)
        arch_str = utils.get_arch_key(child.arch)
        if arch_str not in model_dict:
            try:
                child.accuracy, child.trained_model, child.train_time = utils.train_architecture(child, data)
            except:
                child.accuracy = 0.0
            model_dict[arch_str] = child
            if child.accuracy > best_accuracy:
                best_accuracy = child.accuracy
            print("Architecture: {}\tAccuracy: {}\tCurrent Best: {}\n".format(arch_str, child.accuracy, best_accuracy))
        else:
            child = model_dict[arch_str]
        population.append(child)
        history.append(child)
        population.popleft()

    utils.save_history(history)

    best_model = max(history, key=lambda i: i.accuracy)
    print("Best Architectures: {}\nBest Accuracy: {}".format(best_model.arch, best_model.accuracy))


if __name__ == "__main__":
    args = build_args()
    main(args)
