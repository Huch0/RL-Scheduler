# ── pyg_converter.py ──────────────────────────────────────
from typing import Dict, List, Tuple
import networkx as nx
import torch
from torch_geometric.data import HeteroData

# ── 노드/엣지 인코딩 헬퍼 ────────────────────────────────────
def encode_node(feat: Dict, ntype: str) -> List[float]:
    if ntype == "operation":
        return [
            feat.get("id", 0),
            feat.get("type", 0),
            feat.get("duration", 0),
            feat.get("job_id", 0),
        ]
    elif ntype == "machine":
        return [
            feat.get("id", 0),
            feat.get("queue_len", 0),
            feat.get("busy_until", 0),
        ]
    else:
        return []

def encode_edge(feat: Dict, etype: str) -> List[float]:
    if etype == "assignment":      # 3-dim
        return [
            feat.get("start_time", 0.0),
            feat.get("end_time",   0.0),
            feat.get("repetition", 0.0),
        ]
    elif etype == "completion":    # 1-dim
        return [feat.get("repetition", 0.0)]
    else:                          # no features
        return []

# ── 메인 변환 함수 ─────────────────────────────────────────
def graph_to_heterodata(G: nx.MultiDiGraph) -> HeteroData:
    data = HeteroData()

    # 1) 노드 id 재매핑 (타입별 개별 인덱스)
    op_nodes      = [n for n,d in G.nodes(data=True) if d["ntype"]=="operation"]
    mach_nodes    = [n for n,d in G.nodes(data=True) if d["ntype"]=="machine"]
    op_id_map     = {n:i for i,n in enumerate(op_nodes)}
    mach_id_map   = {n:i for i,n in enumerate(mach_nodes)}

    # 2) 노드 feature 텐서
    data["operation"].x = torch.tensor(
        [encode_node(G.nodes[n]["features"], "operation") for n in op_nodes],
        dtype=torch.float,
    )
    data["machine"].x = torch.tensor(
        [encode_node(G.nodes[n]["features"], "machine") for n in mach_nodes],
        dtype=torch.float,
    )

    # 3) 엣지 관계별 버킷 초기화
    buckets: Dict[Tuple[str,str,str], List[List[int]]] = {}
    eattr  : Dict[Tuple[str,str,str], List[List[float]]] = {}

    def push(src, rel, dst, u, v, attr):
        key = (src, rel, dst)
        buckets.setdefault(key, []).append([u, v])
        if attr is not None:
            eattr.setdefault(key, []).append(attr)

    # 4) 모든 엣지를 타입별로 수집
    for u,v,k,d in G.edges(keys=True, data=True):
        etype = d["etype"]
        if etype=="assignment":
            push("machine","assignment","operation",
                 mach_id_map[u], op_id_map[v],
                 encode_edge(d["features"], "assignment"))
        elif etype=="completion":
            push("operation","completion","operation",
                 op_id_map[u], op_id_map[v],
                 encode_edge(d["features"], "completion"))
        elif etype=="type_valid":
            push("operation","type_valid","machine",
                 op_id_map[u], mach_id_map[v], None)
        elif etype=="logical":
            push("operation","logical","operation",
                 op_id_map[u], op_id_map[v], None)

    # 5) buckets → HeteroData
    for rel, edges in buckets.items():
        src, rel_name, dst = rel
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
        data[src, rel_name, dst].edge_index = ei

        # edge_attr 존재 여부에 따라 저장
        if rel in eattr:
            data[src, rel_name, dst].edge_attr = torch.tensor(
                eattr[rel], dtype=torch.float
            )

    return data
