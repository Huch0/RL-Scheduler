from typing import Dict, List
import networkx as nx
import torch
from torch_geometric.data import Data

# ---------- 2. PyG Data 변환 ----------

# 노드 타입별 필드 정의
def encode_node_feat(feat: Dict, ntype: str) -> List[float]:
    """노드 타입별 feature 인코딩."""
    if ntype == "operation":
        fields = [
            feat.get("id", 0),          # operation id
            feat.get("type", 0),
            feat.get("duration", 0),
            feat.get("job_id", 0),
        ]
    elif ntype == "machine":
        fields = [
            feat.get("id", 0),          # machine id
            feat.get("queue_len", 0),
            feat.get("busy_until", 0),  # 추가: busy_until
            0,                          # padding (operation 노드와 길이 맞추기 위함)
        ]
    else:
        fields = []
    return [float(v) for v in fields]

# 엣지 타입별 feature 정의
def encode_edge_feat(feat: Dict, etype: str) -> List[float]:
    """엣지 타입에 따른 feature 인코딩."""
    if etype == "assignment":
        return [
            float(feat.get("start_time", 0.0)),
            float(feat.get("end_time", 0.0)),
            float(feat.get("repetition", 0.0)),
        ]
    elif etype == "completion":
        return [float(feat.get("repetition", 0.0))]
    else:  # type_valid, logical 등은 feature 없음
        return []

ETYPE2IDX = {"type_valid": 0, "assignment": 1, "logical": 2, "completion": 3}
NTYPE2IDX = {"operation": 0, "machine": 1}

def graph_to_pyg(G: nx.MultiDiGraph) -> Data:
    node_map = {n: i for i, n in enumerate(G.nodes)}
    x, ntypes = [], []

    for n, d in G.nodes(data=True):
        ntype = d["ntype"]
        x.append(encode_node_feat(d["features"], ntype))
        ntypes.append(NTYPE2IDX[ntype])

    edge_idx, edge_attr, etypes = [], [], []
    for u, v, k, d in G.edges(keys=True, data=True):
        etype = d["etype"]
        edge_idx.append([node_map[u], node_map[v]])
        edge_attr.append(encode_edge_feat(d.get("features", {}), etype))
        etypes.append(ETYPE2IDX[etype])

    # padding edge_attr to uniform length
    max_len = max(len(e) for e in edge_attr)
    for e in edge_attr:
        while len(e) < max_len:
            e.append(0.0)

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_idx).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        edge_type=torch.tensor(etypes, dtype=torch.long),
        node_type=torch.tensor(ntypes, dtype=torch.long),
    )
