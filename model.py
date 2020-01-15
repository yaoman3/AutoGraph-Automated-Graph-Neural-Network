import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, add_remaining_self_loops, softmax
from torch_scatter import scatter_add
from search_space import act_map

class GeoLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, att_type="gat", agg_type="sum",\
        concat=True, dropout=0, bias=True, negative_slope=0.2, pool_dim=0):
        if agg_type in ["sum", "mlp"]:
            super().__init__('add')
        elif agg_type in ["mean", "max"]:
            super().__init__(agg_type)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.att_type = att_type
        self.agg_type = agg_type
        self.negative_slope = negative_slope

        # GCN weight
        self.gcn_weight = None

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if self.att_type in ["generalized_linear"]:
            self.general_att_layer = torch.nn.Linear(out_channels, 1, bias=False)

        if self.agg_type in ["mean", "max", "mlp"]:
            if pool_dim <= 0:
                pool_dim = 128
        self.pool_dim = pool_dim
        if pool_dim != 0:
            self.pool_layer = torch.nn.ModuleList()
            self.pool_layer.append(torch.nn.Linear(self.out_channels, self.pool_dim))
            self.pool_layer.append(torch.nn.Linear(self.pool_dim, self.out_channels))
        else:
            pass
        self.reset_parameters()

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

        if self.att_type in ["generalized_linear"]:
            glorot(self.general_att_layer.weight)

        if self.pool_dim != 0:
            for layer in self.pool_layer:
                glorot(layer.weight)
                zeros(layer.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # prepare
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index, x=x, num_nodes=x.size(0), edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_index, num_nodes, edge_weight = None):

        if self.att_type == "const":
            if self.training and self.dropout > 0:
                x_j = F.dropout(x_j, p=self.dropout, training=True)
            neighbor = x_j
        elif self.att_type == "gcn":
            if self.gcn_weight is None or self.gcn_weight.size(0) != x_j.size(0):  # 对于不同的图gcn_weight需要重新计算
                _, norm = self.norm(edge_index, num_nodes, edge_weight)
                self.gcn_weight = norm
            neighbor = self.gcn_weight.view(-1, 1, 1) * x_j
        else:
            # Compute attention coefficients.
            alpha = self.apply_attention(edge_index, num_nodes, x_i, x_j)
            alpha = softmax(alpha, edge_index[0], num_nodes)
            # Sample attention coefficients stochastically.
            if self.training and self.dropout > 0:
                alpha = F.dropout(alpha, p=self.dropout, training=True)

            neighbor = x_j * alpha.view(-1, self.heads, 1)
        if self.pool_dim > 0:
            for layer in self.pool_layer:
                neighbor = layer(neighbor)
        return neighbor

    def apply_attention(self, edge_index, num_nodes, x_i, x_j):
        if self.att_type == "gat":
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)

        elif self.att_type == "gat_sym":
            wl = self.att[:, :, :self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels:]  # weight right
            alpha = (x_i * wl).sum(dim=-1) + (x_j * wr).sum(dim=-1)
            alpha_2 = (x_j * wl).sum(dim=-1) + (x_i * wr).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope) + F.leaky_relu(alpha_2, self.negative_slope)

        elif self.att_type == "linear":
            wl = self.att[:, :, :self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels:]  # weight right
            al = x_j * wl
            ar = x_j * wr
            alpha = al.sum(dim=-1) + ar.sum(dim=-1)
            alpha = torch.tanh(alpha)
        elif self.att_type == "cos":
            wl = self.att[:, :, :self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels:]  # weight right
            alpha = x_i * wl * x_j * wr
            alpha = alpha.sum(dim=-1)

        elif self.att_type == "generalized_linear":
            wl = self.att[:, :, :self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels:]  # weight right
            al = x_i * wl
            ar = x_j * wr
            alpha = al + ar
            alpha = torch.tanh(alpha)
            alpha = self.general_att_layer(alpha)
        else:
            raise Exception("Wrong attention type:", self.att_type)
        return alpha

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def get_param_dict(self):
        params = {}
        key = f"{self.att_type}_{self.agg_type}_{self.in_channels}_{self.out_channels}_{self.heads}"
        weight_key = key + "_weight"
        att_key = key + "_att"
        agg_key = key + "_agg"
        bais_key = key + "_bais"

        params[weight_key] = self.weight
        params[att_key] = self.att
        params[bais_key] = self.bias
        if hasattr(self, "pool_layer"):
            params[agg_key] = self.pool_layer.state_dict()

        return params

    def load_param(self, params):
        key = f"{self.att_type}_{self.agg_type}_{self.in_channels}_{self.out_channels}_{self.heads}"
        weight_key = key + "_weight"
        att_key = key + "_att"
        agg_key = key + "_agg"
        bais_key = key + "_bais"

        if weight_key in params:
            self.weight = params[weight_key]

        if att_key in params:
            self.att = params[att_key]

        if bais_key in params:
            self.bias = params[bais_key]

        if agg_key in params and hasattr(self, "pool_layer"):
            self.pool_layer.load_state_dict(params[agg_key])


class GraphNet(nn.Module):
    def __init__(self, arch, nfeat, nclass, dropout=0.5, state_num=6):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.acts = []
        self.skip = []
        self.build_layers(arch, nfeat, nclass, state_num)

    def build_layers(self, arch, nfeat, nclass, state_num=6):
        assert len(arch) % state_num == 0
        assert arch[-2] == nclass
        nlayer = len(arch) // state_num
        outs = [nfeat]
        for i in range(nlayer):
            if i == 0:
                in_channels = nfeat
            else:
                in_channels = out_channels * head_num
            self.skip.append(arch[i*state_num + 5])
            if self.skip[-1] != -1:
                in_channels += outs[self.skip[-1]]
            attention_type = arch[i*state_num + 0]
            aggregator_type = arch[i*state_num + 1]
            act = arch[i*state_num + 2]
            head_num = arch[i*state_num + 3]
            out_channels = arch[i*state_num + 4]
            outs.append(out_channels * head_num)
            concat = True
            if i == nlayer-1:
                concat = False
            self.layers.append(GeoLayer(in_channels, out_channels, head_num, att_type=attention_type, agg_type=aggregator_type, \
                concat=concat, dropout=self.dropout))
            self.acts.append(act_map(act))

    def forward(self, x, edge_index, edge_weight=None):
        outputs = []
        output = x
        outputs.append(x)
        for i, (act, layer) in enumerate(zip(self.acts, self.layers)):
            if self.skip[i] != -1:
                output = torch.cat((output, outputs[self.skip[i]]), 1)
            if i != 0:
                output = F.dropout(output, p=self.dropout, training=self.training)
            output = layer(output, edge_index, edge_weight)
            output = act(output)
            outputs.append(output)
        return output


class Individual:
    def __init__(self, search_space, architecture=None, nfeat=None, nclass=None):
        self.search_space = search_space
        self.arch = architecture
        self.nfeat = nfeat
        self.nclass = nclass
        self.accuracy = None
        self.model = None
        self.params = None

    def build_gnn(self, nfeat=None, nclass=None):
        # torch.cuda.manual_seed(123)
        if nfeat is None:
            nfeat = self.nfeat
        if nclass is None:
            nclass = self.nclass
        self.model = GraphNet(self.arch, nfeat, nclass)
        return self.model
