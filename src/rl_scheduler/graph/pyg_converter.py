# ── pyg_converter.py ────────────────────────────────────────────
from typing import Dict, List
import networkx as nx
import torch
from torch_geometric.data import Data

# ---------- 2. PyG Data 변환 ----------
def encode_node_feat(feat: Dict) -> List[float]:
    """예시: 간단한 실수 벡터 인코딩 (원-핫/임베딩은 따로)."""
    # build raw feature list
    raw = [
        feat.get("id", 0),
        feat.get("duration", 0),
        feat.get("type", 0),
        feat.get("job_id", 0),
        feat.get("queue_len", 0),
        feat.get("busy_until", 0),
    ]
    # convert all to float, fallback to 0.0 on error
    out = []
    for v in raw:
        try:
            out.append(float(v))
        except Exception:
            out.append(0.0)
    return out

def encode_edge_feat(feat: Dict) -> List[float]:
    # 시간 정보가 없으면 0.0
    return [
        feat.get("start_time", 0.0),
        feat.get("end_time", 0.0),
        feat.get("completion_time", 0.0),
    ]

ETYPE2IDX = {"type_valid": 0, "assignment": 1, "logical": 2, "completion": 3}
NTYPE2IDX = {"operation": 0, "machine": 1}

def graph_to_pyg(G: nx.MultiDiGraph) -> Data:
    node_map = {n: i for i, n in enumerate(G.nodes)}
    x, ntypes = [], []
    for n in G.nodes:
        x.append(encode_node_feat(G.nodes[n]["features"]))
        ntypes.append(NTYPE2IDX[G.nodes[n]["ntype"]])

    edge_idx, edge_attr, etypes = [], [], []
    for u, v, k, d in G.edges(keys=True, data=True):
        edge_idx.append([node_map[u], node_map[v]])
        edge_attr.append(encode_edge_feat(d.get("features", {})))
        etypes.append(ETYPE2IDX[d["etype"]])

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_idx).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        edge_type=torch.tensor(etypes, dtype=torch.long),
        node_type=torch.tensor(ntypes, dtype=torch.long),
    )
