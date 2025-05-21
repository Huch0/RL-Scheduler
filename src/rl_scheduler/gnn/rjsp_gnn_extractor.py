from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from torch_geometric.data import HeteroData

from rl_scheduler.gnn import RJSPGNN
import torch
import torch.nn as nn


class RJSPGraphExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_dim=64, use_global_attention=False):
        super().__init__(observation_space, features_dim=1)

        extractors: dict[str, nn.Module] = {}
        self.rjsp_gnn = RJSPGNN(
            hidden_dim=hidden_dim,
            use_global_attention=use_global_attention,
        )

        total_concat_size = self.rjsp_gnn.out_dim
        for key, subspace in observation_space.spaces.items():
            if key[0:1] == "g_":
                continue
            # if key == "RJSPGraph":
            #     extractors[key] = self.rjsp_gnn
            #     total_concat_size += self.rjsp_gnn.out_dim
            # else:
            # The observation key is a vector, flatten it if needed
            extractors[key] = nn.Flatten()
            total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            if key[0:1] == "g_":
                continue
            encoded_tensor_list.append(extractor(observations[key]))

        # Graph features
        # NumPy → HeteroData
        data = HeteroData()
        data["machine"].x = torch.as_tensor(
            observations["g_machine_x"], dtype=torch.float32
        ).squeeze()
        data["operation"].x = torch.as_tensor(
            observations["g_operation_x"], dtype=torch.float32
        ).squeeze()
        # scatter(): Expected dtype int64 for index
        # index_select(): Expected dtype int32 or int64 for index
        # Expected 'edge_index' to be of integer type
        # Expected 'edge_index' to be two-dimensional
        data["machine", "assignment", "operation"].edge_index = torch.as_tensor(
            observations["g_edge_idx_assignment"], dtype=torch.int64
        ).squeeze()
        data["machine", "assignment", "operation"].edge_attr = torch.as_tensor(
            observations["g_edge_attr_assignment"], dtype=torch.float32
        ).squeeze()
        data["operation", "completion", "operation"].edge_index = torch.as_tensor(
            observations["g_edge_idx_completion"], dtype=torch.int64
        ).squeeze()
        data["operation", "completion", "operation"].edge_attr = torch.as_tensor(
            observations["g_edge_attr_completion"], dtype=torch.float32
        ).squeeze(0)
        data["operation", "type_valid", "machine"].edge_index = torch.as_tensor(
            observations["g_edge_idx_type_valid"], dtype=torch.int64
        ).squeeze()
        data["operation", "logical", "operation"].edge_index = torch.as_tensor(
            observations["g_edge_idx_logical"], dtype=torch.int64
        ).squeeze()
        # gnn forward
        g = self.rjsp_gnn(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        encoded_tensor_list.append(g)

        return torch.cat(encoded_tensor_list, dim=1)
