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

# ───────────────────────── 경로 세팅 ────────────────────────────
CONTRACTS = INSTANCES_DIR / "contracts"  / "C-test0.json"
MACHINES  = INSTANCES_DIR / "machines"   / "M-test0-2.json"
JOBS      = INSTANCES_DIR / "jobs"       / "J-test0-2.json"
OPS       = INSTANCES_DIR / "operations" / "O-test0.json"

OUT_DIR   = Path("./graph_debug_png")
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────── 시각화 유틸 ────────────────────────────
def quick_draw(G: nx.MultiDiGraph, title: str, save_path: Path | None = None):
    pos = nx.spring_layout(G, seed=0)
    node_colors = [
        "skyblue" if G.nodes[n]["ntype"] == "operation" else "lightgreen"
        for n in G.nodes
    ]
    edge_colors = [
        "red"       if d.get("etype") == "completion"
        else "blue" if d.get("etype") == "assignment"
        else "gray" if d.get("etype") == "type_valid"
        else "lightgray"
        for _, _, d in G.edges(data=True)
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    nx.draw(
        G, pos, ax=ax, with_labels=True,
        node_color=node_colors, edge_color=edge_colors,
        node_size=900, font_size=8
    )

    edge_labels = {
        (u, v): d["label"]
        for u, v, d in G.edges(data=True) if "label" in d
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()

    plt.close(fig)

# ──────────────────────── 테스트 케이스 ─────────────────────────
def test_graph_over_steps():
    # 1) Scheduler 초기화
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

    # 2) STEP 0 ─ 초기 그래프
    G0 = build_graph_from_scheduler(sched)
    _  = graph_to_pyg(G0)          # shape 확인만
    quick_draw(G0, "Step 0 (initial)", OUT_DIR / "graph_step0.png")

    # 3) STEP 1  (machine 0, job 0, rep 0)
    sched.step(chosen_machine_id=0, chosen_job_id=0, chosen_repetition=0)
    G1 = build_graph_from_scheduler(sched)
    _  = graph_to_pyg(G1)
    quick_draw(G1, "Step 1  (M0 → J0-R0)", OUT_DIR / "graph_step1.png")

    # 4) STEP 2  (machine 0, job 1, rep 0)
    sched.step(chosen_machine_id=0, chosen_job_id=1, chosen_repetition=0)
    G2 = build_graph_from_scheduler(sched)
    _  = graph_to_pyg(G2)
    quick_draw(G2, "Step 2  (M0 → J1-R0)", OUT_DIR / "graph_step2.png")

    # 간단 무결성 체크 (노드·엣지 수 증가 확인)
    assert G1.number_of_edges() >= G0.number_of_edges()
    assert G2.number_of_edges() >= G1.number_of_edges()
