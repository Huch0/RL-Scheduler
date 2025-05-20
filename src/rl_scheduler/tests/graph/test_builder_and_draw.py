# tests/graph/test_graph_over_steps.py
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

import pytest

from rl_scheduler.scheduler.scheduler import Scheduler
from rl_scheduler.scheduler.slot_allocator import LinearSlotAllocator
from rl_scheduler.contract_generator import DeterministicGenerator
from rl_scheduler.graph.builder import build_graph_from_scheduler
from rl_scheduler.graph.pyg_converter import graph_to_pyg
from rl_scheduler.config_path import INSTANCES_DIR

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
    G0 = build_graph_from_scheduler(sched)
    _  = graph_to_pyg(G0)                # 변환만 (shape-check)
    quick_draw(G0, "Step 0 (initial)", tmp_path / "graph_step0.png")

    # 3) Step 1 : (machine 0, job 0, rep 0)
    sched.step(chosen_machine_id=0, chosen_job_id=0, chosen_repetition=0)
    G1 = build_graph_from_scheduler(sched)
    _  = graph_to_pyg(G1)
    quick_draw(G1, "Step 1  (M0 → J0-R0)", tmp_path / "graph_step1.png")

    # 4) Step 2 : (machine 0, job 1, rep 0)
    sched.step(chosen_machine_id=0, chosen_job_id=1, chosen_repetition=0)
    G2 = build_graph_from_scheduler(sched)
    _  = graph_to_pyg(G2)
    quick_draw(G2, "Step 2  (M0 → J1-R0)", tmp_path / "graph_step2.png")

    # ── 간단 무결성 체크 ─────────────────────────────────────
    assert G1.number_of_edges() >= G0.number_of_edges()
    assert G2.number_of_edges() >= G1.number_of_edges()
    # assignment·completion 엣지가 실제로 추가되었는지
    assert any(d["etype"] == "assignment"  for _, _, d in G1.edges(data=True))
    assert any(d["etype"] == "completion"  for _, _, d in G2.edges(data=True))


# ── PyG 변환 무결성 테스트 ────────────────────────────────────
def test_pyg_conversion_shapes():
    """builder → graph_to_pyg 변환 시 텐서 차원 및 매핑 체크."""
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

    # 한-두 번 step 해서 assignment / completion 엣지 포함시키기
    sched.step(chosen_machine_id=0, chosen_job_id=0, chosen_repetition=0)
    sched.step(chosen_machine_id=0, chosen_job_id=1, chosen_repetition=0)

    G = build_graph_from_scheduler(sched)
    data = graph_to_pyg(G)

    # ── 노드 텐서 ────────────────────────────────────────────
    assert data.x.shape[0] == G.number_of_nodes()          # 행 수 = 노드 수
    assert data.x.shape[1] == 4                            # pad 포함 feature dim
    assert data.node_type.shape[0] == G.number_of_nodes()  # 타입 길이 일치
    # node_type 값이 (0,1) 범위인지
    assert set(data.node_type.tolist()) <= {0, 1}

    # ── 엣지 텐서 ────────────────────────────────────────────
    num_edges = G.number_of_edges()
    assert data.edge_index.shape == (2, num_edges)
    assert data.edge_type.shape[0] == num_edges
    assert data.edge_attr.shape[0] == num_edges
    # edge_attr 는 assignment(3)·completion(1)·others(0)→pad 로 3 차원
    assert data.edge_attr.shape[1] == 3

    # assignment / completion edge_type 인덱스 존재 확인
    assert 1 in data.edge_type.tolist()    # assignment
    assert 3 in data.edge_type.tolist()    # completion