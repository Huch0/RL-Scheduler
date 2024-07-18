import numpy as np
import scipy.signal

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool


class GPPO(nn.Module):
    def __init__(self,
                 # GNN parameters
                 gnn_type='GIN',
                 gnn_kwargs=dict(),
                 input_feature_dim=2,
                 # Actor-Critic parameters
                 actor_hidden_sizes=(64, 64),
                 actor_activation=nn.Tanh,
                 critic_hidden_sizes=(64, 64),
                 critic_activation=nn.Tanh,
                 # shared parameters
                 output_feature_dim=64,
                 device='cpu'
                 ):

        super(GPPO, self).__init__()

        self.feature_extractor = None
        if gnn_type == 'GIN':
            self.feature_extractor = GIN(input_feature_dim=input_feature_dim, hidden_dim=output_feature_dim).to(device)
        else:
            raise NotImplementedError(f'GNN type {gnn_type} not implemented')

        self.pi = Actor(output_feature_dim=output_feature_dim,
                        hidden_sizes=actor_hidden_sizes, activation=actor_activation).to(device)
        self.v = Critic(output_feature_dim=output_feature_dim,
                        hidden_sizes=critic_hidden_sizes, activation=critic_activation).to(device)

    def step(self,
             graph: Data,
             candidate_node_indices: th.Tensor,
             ):

        with th.no_grad():
            # get the node embeddings for each graph in the batch
            h_nodes, h_graph = self.feature_extractor(graph.x, graph.edge_index)

            # get the candidate node embeddings
            candidate_node_embeddings = h_nodes[candidate_node_indices]

            # get the action distribution
            pi = self.pi._distribution(h_graph, candidate_node_embeddings)
            a_index = pi.sample()
            a = candidate_node_indices[a_index]
            logp_a = self.pi._log_prob_from_distribution(pi, a_index)
            v = self.v(h_graph)

        return a, v, logp_a

    def act(self, graph: Data, candidate_node_indices: th.Tensor):
        return self.step(graph, candidate_node_indices)[0]

    def forward(self, graph: Batch, candidate_node_indices: th.Tensor, action: th.Tensor):
        # get the node embeddings for each graph in the batch
        h_nodes, h_graph = self.feature_extractor(graph.x, graph.edge_index, graph.batch)

        pis = []
        logp_as = []
        for i, can in enumerate(candidate_node_indices):
            # get the candidate node embeddings
            candidate_node_embeddings = h_nodes[can]

            # get the action distribution
            pi = self.pi._distribution(h_graph[i], candidate_node_embeddings)
            a_index = th.where(candidate_node_indices[i] == action[i])[0]
            logp_a = self.pi._log_prob_from_distribution(pi, a_index)

            pis.append(pi)
            logp_as.append(logp_a)

        # Convert to 1d tensor
        logp_as = th.stack(logp_as).flatten()

        return pis, logp_as
    
    def compute_v(self, graph: Batch):
        h_nodes, h_graph = self.feature_extractor(graph.x, graph.edge_index, graph.batch)
        return self.v(h_graph)



class GIN(nn.Module):
    def __init__(self, input_feature_dim, num_layers=2, hidden_dim=64):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(
            GINConv(
                nn.Sequential(nn.Linear(input_feature_dim, hidden_dim),
                              th.nn.BatchNorm1d(hidden_dim),
                              nn.ReLU(),
                              nn.Linear(hidden_dim, hidden_dim)
                              )
            )
        )

        for _ in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                  th.nn.BatchNorm1d(hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim)
                                  )
                )
            )

    def forward(self, x, edge_index, batch = None):
        h = self.convs[0](x, edge_index)
        for conv in self.convs[1:]:
            h = conv(h, edge_index)
        # average pooling
            h_graph = global_mean_pool(h, batch)

        return h, h_graph

class Actor(nn.Module):
    def __init__(self,
                 output_feature_dim: int,
                 hidden_sizes: tuple,
                 activation: nn.Module,
                 ):
        super(Actor, self).__init__()

        self.logits_net = mlp([output_feature_dim * 2] + list(hidden_sizes) + [1], activation)

    def _distribution(self, h_graph, candidate_node_embeddings):
        h_graph = h_graph.repeat(candidate_node_embeddings.shape[0], 1)
        logits = self.logits_net(th.cat([h_graph, candidate_node_embeddings], dim=-1))
        logits = logits.flatten()
        return th.distributions.Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class Critic(nn.Module):
    def __init__(self,
                 output_feature_dim: int,
                 hidden_sizes: tuple,
                 activation: nn.Module,
                 ):
        super(Critic, self).__init__()

        self.v_net = mlp([output_feature_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, h_graph):
        return self.v_net(h_graph)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
