# tests/graph/test_graph_over_steps.py
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

import pytest

from rl_scheduler.scheduler.scheduler import Scheduler
from rl_scheduler.scheduler.slot_allocator import LinearSlotAllocator
from rl_scheduler.contract_generator import DeterministicGenerator
from rl_scheduler.graph.pyg_converter import graph_to_heterodata
from rl_scheduler.config_path import INSTANCES_DIR
from torch_geometric.data import HeteroData


# ── 테스트 인스턴스 경로 ────────────────────────────────────────
CONTRACTS = INSTANCES_DIR / "contracts"  / "C-test0.json"
MACHINES  = INSTANCES_DIR / "machines"   / "M-test0-2.json"
JOBS      = INSTANCES_DIR / "jobs"       / "J-test0-2.json"
OPS       = INSTANCES_DIR / "operations" / "O-test0.json"

OUT_DIR = Path("./graph_debug_png")
OUT_DIR.mkdir(exist_ok=True)

# ── 간단 시각화 함수 ───────────────────────────────────────────
def quick_draw(G: nx.MultiDiGraph, title: str, path: Path | None = None) -> None:
    pos = nx.spring_layout(G, seed=0)

    node_colors = [
        "skyblue"   if G.nodes[n]["ntype"] == "operation" else "lightgreen"
        for n in G.nodes
    ]
    edge_colors = [
        "red"       if d["etype"] == "completion"
        else "blue" if d["etype"] == "assignment"
        else "gray" if d["etype"] == "type_valid"
        else "lightgray"
        for _, _, d in G.edges(data=True)
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    nx.draw(
        G, pos, ax=ax,
        with_labels=True, node_color=node_colors, edge_color=edge_colors,
        node_size=900, font_size=7
    )
    edge_labels = {
        (u, v): d["label"]
        for u, v, d in G.edges(data=True) if "label" in d
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    ax.set_title(title, fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    plt.close(fig)

# ── 메인 테스트 ───────────────────────────────────────────────
def test_graph_over_steps(tmp_path: Path = OUT_DIR):
    """Build → step() 2회 → 그래프/PyG 변환 및 무결성 확인."""
    # 1) scheduler 초기화
    det = DeterministicGenerator(CONTRACTS)
    reps  = det.load_repetition()
    profs = det.load_profit_fn()

    sched = Scheduler(
        machine_config_path=MACHINES,
        job_config_path=JOBS,
        operation_config_path=OPS,
        slot_allocator=LinearSlotAllocator,
    )
    sched.reset(reps, profs)

    # 2) Step 0 : 초기 그래프
    G0 = sched.graph_sync.G  # 스케줄러의 그래프 동기화 객체에서 직접 그래프 접근
    _  = graph_to_heterodata(G0)                # 변환만 (shape-check)
    quick_draw(G0, "Step 0 (initial)", tmp_path / "graph_step0.png")

    # 3) Step 1 : (machine 0, job 0, rep 0)
    sched.step(chosen_machine_id=0, chosen_job_id=0, chosen_repetition=0)
    G1 = sched.graph_sync.G  # 업데이트된 그래프 직접 접근
    _  = graph_to_heterodata(G1)
    quick_draw(G1, "Step 1  (M0 → J0-R0)", tmp_path / "graph_step1.png")

    # 4) Step 2 : (machine 0, job 1, rep 0)
    sched.step(chosen_machine_id=0, chosen_job_id=1, chosen_repetition=0)
    G2 = sched.graph_sync.G  # 업데이트된 그래프 직접 접근
    _  = graph_to_heterodata(G2)
    quick_draw(G2, "Step 2  (M0 → J1-R0)", tmp_path / "graph_step2.png")

    # ── 간단 무결성 체크 ─────────────────────────────────────
    assert G1.number_of_edges() >= G0.number_of_edges()
    assert G2.number_of_edges() >= G1.number_of_edges()
    # assignment·completion 엣지가 실제로 추가되었는지
    assert any(d["etype"] == "assignment"  for _, _, d in G1.edges(data=True))
    assert any(d["etype"] == "completion"  for _, _, d in G2.edges(data=True))

# ── HeteroData 변환 무결성 테스트 ────────────────────────────
def test_heterodata_conversion_shapes():
    """Builder → HeteroData 변환 후 노드/엣지 스키마 확인."""
    det   = DeterministicGenerator(CONTRACTS)
    reps  = det.load_repetition()
    profs = det.load_profit_fn()

    sched = Scheduler(
        machine_config_path=MACHINES,
        job_config_path=JOBS,
        operation_config_path=OPS,
        slot_allocator=LinearSlotAllocator,
    )
    sched.reset(reps, profs)

    # 두 번 step 하여 assignment·completion 엣지 생성
    sched.step(chosen_machine_id=0, chosen_job_id=0, chosen_repetition=0)
    sched.step(chosen_machine_id=0, chosen_job_id=1, chosen_repetition=0)

    G = sched.graph_sync.G  # 스케줄러의 그래프 동기화 객체에서 직접 그래프 접근
    data: HeteroData = graph_to_heterodata(G)              # ★ 변경

    # ── 노드 검증 ──────────────────────────────────────────
    assert data["machine"].x.size(1)   == 3    # id, queue_len, busy_until
    assert data["operation"].x.size(1) == 4    # id, type, dur, job_id

    # 노드 수 일치
    assert data["machine"].num_nodes + data["operation"].num_nodes == G.number_of_nodes()

    # ── 엣지 검증 (관계별) ───────────────────────────────────
    # ① assignment : machine → operation
    a_rel = ("machine", "assignment", "operation")
    assert a_rel in data.edge_index_dict
    assert data[a_rel].edge_attr.size(1) == 3   # start, end, rep

    # ② completion : operation → operation
    c_rel = ("operation", "completion", "operation")
    assert c_rel in data.edge_index_dict
    assert data[c_rel].edge_attr.size(1) == 1   # rep only

    # ③ type_valid : operation → machine (no edge_attr)
    tv_rel = ("operation", "type_valid", "machine")
    assert tv_rel in data.edge_index_dict
    assert not hasattr(data[tv_rel], "edge_attr")

    # ④ logical : operation ↔ operation (bidirectional, no edge_attr)
    log_rel = ("operation", "logical", "operation")
    assert log_rel in data.edge_index_dict
    src, dst = data[log_rel].edge_index
    # 각 (u,v)에 대해 (v,u) 가 존재 → 집합 비교
    edge_set = {(int(s), int(d)) for s, d in zip(src.tolist(), dst.tolist())}
    for s, d in edge_set:
        assert (d, s) in edge_set

    # 전체 엣지 수 일치
    hetero_edge_total = sum(
        ei.size(1) for ei in data.edge_index_dict.values()
    )
    assert hetero_edge_total == G.number_of_edges()
