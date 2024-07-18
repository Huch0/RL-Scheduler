import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, Sequential, global_mean_pool


class GPPO(nn.Module):
    def __init__(self,
                 # GNN parameters
                 gnn_type='GIN',
                 gnn_kwargs=dict(),
                 # Actor-Critic parameters
                 actor_hidden_sizes=(64, 64),
                 actor_activation=nn.Tanh,
                 critic_hidden_sizes=(64, 64),
                 critic_activation=nn.Tanh,
                 # shared parameters
                 node_feature_space=(2,),
                 device='cpu'
                 ):

        super(GPPO, self).__init__()

        self.feature_extractor = None
        if gnn_type == 'GIN':
            self.feature_extractor = GIN(node_feature_space=node_feature_space, **gnn_kwargs).to(device)
        else:
            raise NotImplementedError(f'GNN type {gnn_type} not implemented')

        self.pi = Actor(node_feature_space=self.feature_extractor.node_feature_space,
                           actor_hidden_sizes=actor_hidden_sizes, actor_activation=actor_activation).to(device)
        self.v = Critic(node_feature_space=self.feature_extractor.node_feature_space,
                             critic_hidden_sizes=critic_hidden_sizes, critic_activation=critic_activation).to(device)

    def step(self,
             graph_batch: Batch,
             candidate_node_indices: th.Tensor,
             ):
        
        """_summary_
            graph_batch: a batch of graphs (torch_geometric.data.Batch) (batch_size,)
            candidate_node_indices: the indices of the candidate nodes for each graph in the batch (torch.Tensor) (batch_size, num_candidates)
        """

        with th.no_grad():
            # get the node embeddings for each graph in the batch
            h_nodes, h_graph = self.feature_extractor(graph_batch.x, graph_batch.edge_index)

            # get the candidate node embeddings
            candidate_node_embeddings = h_nodes[candidate_node_indices] # h_nodes가 batch라면, candidate_node_indices는 (batch_size, num_candidates)이므로, 이렇게 indexing하면 (batch_size, num_candidates, node_feature_space)가 됨

            # get the action distribution
            pi = self.actor._distribution(h_graph, candidate_node_embeddings)
            a = pi.sample()
            logp_a = self.actor._log_prob_from_distribution(pi, a)
            v = self.critic(h_graph, candidate_node_embeddings)

        return a, v, logp_a

    def act(self, graph_batch: Batch, candidate_node_indices: th.Tensor):
        return self.step(graph_batch, candidate_node_indices)[0]


class GIN(nn.Module):
    def __init__(self, node_feature_space, num_layers=2, hidden_dim=64):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            mlp = Sequential('x, edge_index', [
                (nn.Linear(hidden_dim if i > 0 else node_feature_space, hidden_dim), 'x -> x'),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ])

            self.convs.append(GINConv(mlp))

        # Optional: Batch normalization
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x) if self.batch_norms else x
            x = F.relu(x)
        # average pooling
        h_graph = global_mean_pool(x, batch)
        return x, h_graph


class Actor(nn.Module):
    def __init__(self,
                 node_feature_space: tuple,
                 hidden_sizes: tuple,
                 activation: nn.Module,
                 ):
        super(Actor, self).__init__()

        self.logits_net = mlp([sum(node_feature_space) * 2] + list(hidden_sizes) + [1], activation)

    def _distribution(self, h_graph, candidate_node_embeddings):
        logits = self.logits_net(th.cat([h_graph, candidate_node_embeddings], dim=-1))
        return th.distributions.Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class Critic(nn.Module):
    def __init__(self,
                 node_feature_space: tuple,
                 hidden_sizes: tuple,
                 activation: nn.Module,
                 ):
        super(Critic, self).__init__()

        self.v_net = mlp([sum(node_feature_space) * 2] + list(hidden_sizes) + [1], activation)

    def forward(self, h_graph, candidate_node_embeddings):
        return self.v_net(th.cat([h_graph, candidate_node_embeddings], dim=-1))


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)
