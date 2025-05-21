import torch
from torch_geometric.nn import HeteroConv, GINEConv, SAGEConv
from torch_geometric.nn import Linear
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import AttentionalAggregation

NUM_FEATURES_MACHINE = 3
NUM_FEATURES_OPERATION = 4
NUM_FEATURES_ASSIGNMENT = 3
NUM_FEATURES_COMPLETION = 1


class RJSPGNN(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        num_features_machine=NUM_FEATURES_MACHINE,
        num_features_operation=NUM_FEATURES_OPERATION,
        num_features_assignment=NUM_FEATURES_ASSIGNMENT,
        num_features_completion=NUM_FEATURES_COMPLETION,
        use_global_attention=False,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.num_features_machine = num_features_machine
        self.num_features_operation = num_features_operation
        self.num_features_assignment = num_features_assignment
        self.num_features_completion = num_features_completion
        self.hidden_dim = hidden_dim

        # Node Encoders
        self.machine_encoder = Linear(num_features_machine, hidden_dim)
        self.operation_encoder = Linear(num_features_operation, hidden_dim)

        # Message Passing Layers
        self.convs.append(
            HeteroConv(
                {
                    ("machine", "assignment", "operation"): GINEConv(
                        nn=Linear(hidden_dim, hidden_dim),
                        edge_dim=num_features_assignment,
                    ),
                    ("operation", "completion", "operation"): GINEConv(
                        nn=Linear(hidden_dim, hidden_dim),
                        edge_dim=num_features_completion,
                    ),
                    ("operation", "type_valid", "machine"): SAGEConv(
                        (-1, -1), hidden_dim
                    ),
                    ("operation", "logical", "operation"): SAGEConv(
                        (-1, -1), hidden_dim
                    ),
                },
                aggr="mean",
            )
        )

        # Global Attention Layer (if needed)
        self.use_global_attention = use_global_attention
        if self.use_global_attention:
            self.global_machine_attention = AttentionalAggregation(
                gate_nn=Linear(hidden_dim, hidden_dim)
            )
            self.global_operation_attention = AttentionalAggregation(
                gate_nn=Linear(hidden_dim, hidden_dim)
            )

        self.out_dim = 2 * hidden_dim

    # --- helper: run on a *single* graph ------------------------------ #
    def _forward_single(self, x_dict, edge_index_dict, edge_attr_dict):
        # 1) Encode
        x_dict["machine"] = self.machine_encoder(x_dict["machine"])
        x_dict["operation"] = self.operation_encoder(x_dict["operation"])

        # 2) zero‑fill edge_attr if needed
        if edge_attr_dict is None:
            edge_attr_dict = {}
        for rel, dim in {
            ("machine", "assignment", "operation"): self.num_features_assignment,
            ("operation", "completion", "operation"): self.num_features_completion,
        }.items():
            if rel in edge_index_dict and rel not in edge_attr_dict:
                num_e = edge_index_dict[rel].size(1)
                edge_attr_dict[rel] = torch.zeros(
                    (num_e, dim),
                    dtype=x_dict["machine"].dtype,
                    device=edge_index_dict[rel].device,
                )
        # 3) message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        # 4) global pooling (single graph → mean/attention)
        if self.use_global_attention:
            g_m = self.global_machine_attention(x_dict["machine"])
            g_o = self.global_operation_attention(x_dict["operation"])
        else:
            g_m = x_dict["machine"].mean(dim=0, keepdim=True)
            g_o = x_dict["operation"].mean(dim=0, keepdim=True)
        return torch.cat([g_m, g_o], dim=-1)  # [1, 2*h]

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # Detect batched input (3‑dim tensors)
        if x_dict["machine"].dim() == 3:
            B = x_dict["machine"].size(0)
            g_list = []
            for b in range(B):
                sub_x = {k: v[b] for k, v in x_dict.items()}
                sub_ei = {k: v[b] for k, v in edge_index_dict.items()}
                sub_ea = (
                    {k: v[b] for k, v in edge_attr_dict.items()}
                    if edge_attr_dict is not None
                    else None
                )
                g_list.append(self._forward_single(sub_x, sub_ei, sub_ea))
            return torch.cat(g_list, dim=0)  # [B, 2*h]

        # Non‑batched path
        return self._forward_single(x_dict, edge_index_dict, edge_attr_dict)

    def check_validity(self, x_dict, edge_index_dict, edge_attr_dict):
        required_node_types = ["machine", "operation"]
        required_edge_types = [
            ("machine", "assignment", "operation"),
            ("operation", "completion", "operation"),
            ("operation", "type_valid", "machine"),
            ("operation", "logical", "operation"),
        ]

        # Node types should be exactly as required
        if set(x_dict.keys()) != set(required_node_types):
            raise ValueError(f"Node types in x_dict should be {required_node_types}.")

        # Edge types should be exactly as required
        if set(edge_index_dict.keys()) != set(required_edge_types):
            raise ValueError(
                f"Edge types in edge_index_dict should be " f"{required_edge_types}."
            )

        # assignment / completion may be absent; if present, must match dim
        for rel, dim in [
            (("machine", "assignment", "operation"), self.num_features_assignment),
            (("operation", "completion", "operation"), self.num_features_completion),
        ]:
            if rel in edge_index_dict:
                if rel in edge_attr_dict and edge_attr_dict[rel] is not None:
                    if edge_attr_dict[rel].size(1) != dim:
                        raise ValueError(f"Edge_attr dim mismatch for {rel}.")

        # Type_valid edges should have no features
        if edge_attr_dict.get(("operation", "type_valid", "machine")) is not None:
            raise ValueError("Type_valid edges should not have features.")

        # Logical edges should have no features
        if edge_attr_dict.get(("operation", "logical", "operation")) is not None:
            raise ValueError("Logical edges should not have features.")

        # Logical edges should be bidirectional
        logical_edge_index = edge_index_dict[("operation", "logical", "operation")]
        if logical_edge_index.dim() != 2 or logical_edge_index.size(0) != 2:
            raise ValueError("Logical edge_index must be of shape [2, num_edges].")

        # Convert to sets of tuples for easy membership checking
        src, dst = logical_edge_index[0], logical_edge_index[1]
        edge_pairs = {(int(s), int(d)) for s, d in zip(src.tolist(), dst.tolist())}
        for s, d in edge_pairs:
            if (d, s) not in edge_pairs:
                raise ValueError(
                    "Logical edges must be bidirectional: "
                    f"missing reverse edge ({d} -> {s}) for ({s} -> {d})."
                )

        print("Graph structure is valid.")

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.convs})"


if __name__ == "__main__":
    # Example Heterogeneous graph data
    data = HeteroData()

    num_machines = 5
    num_operations = 10
    num_edges_assignment = 3
    num_edges_completion = 4
    num_edges_type_valid = 10
    num_edges_logical = ((num_operations // num_machines) - 1) * num_machines

    # Nodes
    # [num_machines, num_features_machine]
    data["machine"].x = torch.randn(num_machines, NUM_FEATURES_MACHINE)
    # [num_operations, num_features_operation]
    data["operation"].x = torch.randn(num_operations, NUM_FEATURES_OPERATION)

    # Edges
    # Assignment
    # [2, num_edges_assignment]
    assign_src = torch.randint(0, num_machines, (1, num_edges_assignment))
    assign_dst = torch.randint(
        0,
        num_operations,
        (
            1,
            num_edges_assignment,
        ),
    )
    edge_index_assignment = torch.cat([assign_src, assign_dst], dim=0)
    data["machine", "assignment", "operation"].edge_index = edge_index_assignment
    # [num_edges_assignment, num_features_assignment]
    data["machine", "assignment", "operation"].edge_attr = torch.randn(
        num_edges_assignment, NUM_FEATURES_ASSIGNMENT
    )

    # Completion
    # [2, num_edges_completion]
    comp_src = torch.randint(0, num_operations, (1, num_edges_completion))
    comp_dst = torch.randint(
        0,
        num_operations,
        (
            1,
            num_edges_completion,
        ),
    )
    edge_index_completion = torch.cat([comp_src, comp_dst], dim=0)
    data["operation", "completion", "operation"].edge_index = edge_index_completion
    # [num_edges_completion, num_features_completion]
    data["operation", "completion", "operation"].edge_attr = torch.randn(
        num_edges_completion, NUM_FEATURES_COMPLETION
    )

    # Type-valid
    # no features
    # [2, num_edges_type_valid]
    type_valid_src = torch.randint(0, num_operations, (1, num_edges_type_valid))
    type_valid_dst = torch.randint(
        0,
        num_machines,
        (
            1,
            num_edges_type_valid,
        ),
    )
    edge_index_type_valid = torch.cat([type_valid_src, type_valid_dst], dim=0)
    data["operation", "type_valid", "machine"].edge_index = edge_index_type_valid

    # Logical
    # no features and unidirectional
    # [2, num_edges_type_valid]
    logical_src = torch.randint(0, num_operations, (1, num_edges_logical // 2))
    logical_dst = torch.randint(0, num_operations, (1, num_edges_logical // 2))
    edge_index_logical = torch.cat([logical_src, logical_dst], dim=0)
    # Duplicate the edges to make it bidirectional
    edge_index_logical = torch.cat(
        [edge_index_logical, edge_index_logical.flip(0)], dim=1
    )
    data["operation", "logical", "operation"].edge_index = edge_index_logical

    print(f"Data: {data}")

    # Initialize the model
    model = RJSPGNN(hidden_dim=64, use_global_attention=True)
    model.check_validity(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

    # Forward pass
    g = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
    # print(f"Output x_dict shapes: {[x.shape for x in out_x_dict.values()]}")
    print(f"Global embedding g shape: {g.shape}")
