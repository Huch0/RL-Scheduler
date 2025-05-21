from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.rl_scheduler.gnn import RJSPGNN
from src.rl_scheduler.graph.pad_utils import padded_to_heterodata
import torch, torch.nn as nn

class RJSPExt(BaseFeaturesExtractor):
    def __init__(self, obs_space, hidden_dim=64):
        flat_dim  = obs_space["flat"].shape[0]
        super().__init__(obs_space, features_dim=flat_dim + 2*hidden_dim)

        self.flat_dim  = flat_dim
        self.gnn = RJSPGNN(hidden_dim)

        # padding 파라미터 (env 와 동일해야 함)
        # TODO : 이거 파라미터로 바꿔야 함
        self.max_nodes = 64
        self.max_e_asg = 128
        self.max_e_cmp = 128

    def forward(self, obs):
        flat  = obs["flat"]                               # (B, flat_dim)
        graph = obs["graph"]                              # (B, padded_len)

        # 배치 단위로 HeteroData 변환
        hetero = padded_to_heterodata(graph,
                                      self.max_nodes,
                                      self.max_e_asg,
                                      self.max_e_cmp,
                                      device=flat.device)
        _, g = self.gnn(hetero.x_dict,
                        hetero.edge_index_dict,
                        hetero.edge_attr_dict)            # (B, 2*hidden)
        return torch.cat([flat, g], dim=-1)
