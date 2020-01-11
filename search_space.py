import torch
import torch.nn.functional as F

class MacroSearchSpace:
    def __init__(self):
        super().__init__()
        self.search_space = {
            "attention_type": ["gat", "gcn", "cos", "const", "gat_sym", 'linear', 'generalized_linear'],
            'aggregator_type': ["sum", "mean", "max", "mlp", ],
            'activate_function': ["sigmoid", "tanh", "relu", "linear",
                                  "softplus", "leaky_relu", "relu6", "elu"],
            'number_of_heads': [1, 2, 4],
            'hidden_units': [16, 32, 64, 128, 256],
        }

    def get_search_space(self):
        return self.search_space


def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return F.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return F.relu
    elif act == "relu6":
        return F.relu6
    elif act == "softplus":
        return F.softplus
    elif act == "leaky_relu":
        return F.leaky_relu
    else:
        raise Exception("Unkown activate function")
