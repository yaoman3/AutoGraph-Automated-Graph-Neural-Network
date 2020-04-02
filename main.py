import argparse
import time
import torch.multiprocessing as tmp
import collections
import random
import torch
from search_space import MacroSearchSpace
from model import Individual
import ast
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
    parser.add_argument('--search_layers', type=int, default=3)
    parser.add_argument('--num_processes', type=int, default=4)
    parser.add_argument('--update_batch', type=int, default=12)
    # child model
    parser.add_argument("--dataset", type=str, default="R8", required=False,
                        help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument("--early_stop", type=int, default=60,
                        help="early stop epochs")
    parser.add_argument("--max_evals", type=int, default=60,
                        help="hyperopt evaluation number")
    parser.add_argument("--in-drop", type=float, default=0.5,
                        help="input feature dropout")
    parser.add_argument("--hyperopt", action="store_true",
                        help="do hyperparameter search")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--test_structure', type=str, default="")
    parser.add_argument('--model_dict', type=str, default="")


def main(args): 
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    if args.search_space == "MacroSearchSpace":
        search_space = MacroSearchSpace().get_search_space()

    data, nfeat, nclass = utils.load_data(args.dataset, 'cpu')

    if args.mode == 'test':
        candidate = Individual(search_space, nfeat=nfeat, nclass=nclass)
        candidate.arch = ast.literal_eval(args.test_structure)
        utils.eval_architecture(candidate, data, args.lr, args.weight_decay, args.epochs, args.early_stop, args.model_dict)
    else:
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
        # best_model = None
        mp = tmp.get_context('spawn')
        pool = mp.Pool(args.num_processes, maxtasksperchild=1)
        manager = mp.Manager()
        lock = manager.Lock()
        cuda_dict = manager.dict()
        for i in range(args.num_processes):
            cuda_dict[i] = False
        results = []
        for candidate in population:
            arch_str = utils.get_arch_key(candidate.arch)
            if arch_str not in model_dict:
                # utils.train_architecture(candidate, data, cuda_dict, lock, args.epochs, args.early_stop, args.max_evals)
                result = pool.apply_async(utils.train_architecture, (candidate, data, cuda_dict, lock, args.epochs, args.early_stop, args.max_evals, args.lr, args.weight_decay, args.hyperopt))
                model_dict[arch_str] = candidate
                # try:
                #     candidate.accuracy, candidate.train_time = utils.train_architecture(candidate, data)
                # except:
                #     candidate.accuracy = 0.0
                # model_dict[arch_str] = candidate
                # if candidate.accuracy > best_accuracy:
                #     best_accuracy = candidate.accuracy
                # print("Architecture: {}\tAccuracy: {}\tCurrent Best: {}".format(arch_str, candidate.accuracy, best_accuracy))
            else:
                result = model_dict[arch_str]
                # candidate.accuracy = model_dict[arch_str].accuracy
                # candidate.train_time = model_dict[arch_str].train_time
            results.append(result)
        generation_num = 0
        candidate_num = 0
        for candidate, result in zip(population, results):
            arch_str = utils.get_arch_key(candidate.arch)
            if isinstance(result, Individual):
                candidate.accuracy = result.accuracy
                candidate.train_time = result.train_time
                candidate.params = result.params
            else:
                result.wait()
                candidate.accuracy, candidate.train_time, candidate.params = result.get()
                if candidate.accuracy>best_accuracy:
                    best_accuracy = candidate.accuracy
                    # best_model = model_params_dict
            print("ID: {}\tArchitecture: {}\tAccuracy: {}\tCurrent Best: {}".format(candidate_num, arch_str, candidate.accuracy, best_accuracy))
            candidate_num += 1
            history.append(candidate)
        history_file = utils.save_history(history)

        while len(history)<args.evolution_size:
            childs = []
            results = []
            for _ in range(args.update_batch):
                sample = random.choices(list(population), k = args.sample_size)
                parent = max(sample, key=lambda i: i.accuracy)
                child = Individual(search_space, nfeat=nfeat, nclass=nclass)
                child.arch = utils.mutate_arch(parent)
                arch_str = utils.get_arch_key(child.arch)
                if arch_str not in model_dict:
                    result = pool.apply_async(utils.train_architecture, (child, data, cuda_dict, lock, args.epochs, args.early_stop, args.max_evals, args.lr, args.weight_decay, args.hyperopt))
                    model_dict[arch_str] = child
                else:
                    result = model_dict[arch_str]
                childs.append(child)
                results.append(result)
            for child, result in zip(childs, results):
                arch_str = utils.get_arch_key(child.arch)
                if isinstance(result, Individual):
                    child.accuracy = result.accuracy
                    child.train_time = result.train_time
                    child.params = result.params
                else:
                    try:
                        result.wait(3600)
                    except:
                        print("Timeout for :{}".format(arch_str))
                        continue
                    child.accuracy, child.train_time, child.params = result.get()
                    if child.accuracy>best_accuracy:
                        best_accuracy = child.accuracy
                        # best_model = model_params_dict
                print("ID: {}\tArchitecture: {}\tAccuracy: {}\tCurrent Best: {}".format(candidate_num, arch_str, child.accuracy, best_accuracy))
                candidate_num += 1
                population.append(child)
                history.append(child)
                population.popleft()
            generation_num += 1
            utils.save_history(history[-args.population_size:], history_file, generation_num)

        # utils.save_model(best_model, history_file)
        history_file.close()
        best_individual = max(history, key=lambda i: i.accuracy)
        print("Best Architectures: {}\nBest Accuracy: {}".format(best_individual.arch, best_individual.accuracy))


if __name__ == "__main__":
    args = build_args()
    main(args)
