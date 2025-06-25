# ── pyg_converter.py ──────────────────────────────────────
from typing import Dict, List, Tuple
import networkx as nx, torch
from torch_geometric.data import HeteroData

# -------------------- helpers ----------------------------
def encode_node(feat: Dict, ntype: str) -> List[float]:
    if ntype == "operation":           # 4-dim
        return [
            feat.get("id", 0),
            feat.get("type", 0),
            feat.get("duration", 0),
            feat.get("job_id", 0),
        ]
    if ntype == "machine":             # 3-dim
        return [
            feat.get("id", 0),
            feat.get("queue_len", 0),
            feat.get("busy_until", 0),
        ]
    return []

def encode_edge(feat: Dict, etype: str) -> List[float]:
    if etype == "assignment":          # [start, end, rep]
        return [
            feat.get("start_time", 0.0),
            feat.get("end_time",   0.0),
            feat.get("repetition", 0.0),
        ]
    if etype == "completion":          # [rep]  (≡ job_id fallback)
        return [feat.get("repetition", feat.get("job_instance_id", 0.0))]
    return []                          # type_valid / logical

# -------------------- main -------------------------------
def graph_to_heterodata(G: nx.MultiDiGraph) -> HeteroData:
    data = HeteroData()

    # 1) 타입별 id 재매핑 ---------------------------------------------------
    ops   = [n for n, d in G.nodes(data=True) if d["ntype"] == "operation"]
    machs = [n for n, d in G.nodes(data=True) if d["ntype"] == "machine"]
    op_id   = {n: i for i, n in enumerate(ops)}
    mach_id = {n: i for i, n in enumerate(machs)}

    # 2) 노드 feature -------------------------------------------------------
    data["operation"].x = torch.tensor(
        [encode_node(G.nodes[n]["features"], "operation") for n in ops],
        dtype=torch.float,
    )
    data["machine"].x = torch.tensor(
        [encode_node(G.nodes[n]["features"], "machine") for n in machs],
        dtype=torch.float,
    )

    # 3) relation bucket ----------------------------------------------------
    buckets: dict[Tuple[str, str, str], list[list[int]]] = {}
    eattrs : dict[Tuple[str, str, str], list[list[float]]] = {}

    def push(src_t, rel, dst_t, u_raw, v_raw, attr):
        try:
            u = op_id[u_raw]   if src_t == "operation" else mach_id[u_raw]
            v = op_id[v_raw]   if dst_t == "operation" else mach_id[v_raw]
        except KeyError:       # 방어적 skip (예: mapping 실패)
            return
        key = (src_t, rel, dst_t)
        buckets.setdefault(key, []).append([u, v])
        if attr is not None:
            eattrs.setdefault(key, []).append(attr)

    # 4) 엣지 수집 ----------------------------------------------------------
    for u, v, _, d in G.edges(keys=True, data=True):
        et = d["etype"]
        if et == "assignment":
            push("machine", "assignment", "operation",
                 u, v, encode_edge(d["features"], et))
        elif et == "completion":
            push("operation", "completion", "operation",
                 u, v, encode_edge(d["features"], et))
        elif et == "type_valid":
            push("operation", "type_valid", "machine", u, v, None)
        elif et == "logical":
            push("operation", "logical", "operation", u, v, None)

    # 5) HeteroData 채우기 --------------------------------------------------
    for rel, edge_list in buckets.items():
        src, rel_name, dst = rel
        ei = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        data[(src, rel_name, dst)].edge_index = ei
        if rel in eattrs:                                    # feature 존재
            data[(src, rel_name, dst)].edge_attr = torch.tensor(
                eattrs[rel], dtype=torch.float
            )

    return data
