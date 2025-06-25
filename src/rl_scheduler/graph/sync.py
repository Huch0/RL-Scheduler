# ── sync.py ──────────────────────────────────────────────
import networkx as nx
from typing import Optional
from .builder import build_graph_from_scheduler  # import graph builder
from datetime import datetime

class SchedulerGraphSync:
    """Keeps a live NetworkX MultiDiGraph in sync with Scheduler state."""
    def __init__(self, scheduler):
        self.sch = scheduler
        self.G   = build_graph_from_scheduler(scheduler)  # 첫 스냅숏

    # ---------- 호출 지점 ①: Scheduler.step 직후 ----------
    def on_assignment(self, machine, op):
        """Update graph when an operation is just scheduled."""
        m_node = f"mach_{machine.machine_template.machine_template_id}"
        o_node = f"op_{op.operation_template.operation_template_id}"

        # (1) 머신 노드 실시간 상태 갱신
        m_feat = self.G.nodes[m_node]["features"]
        m_feat["queue_len"]   = len(machine.assigned_operations)
        m_feat["busy_until"]  = machine.last_assigned_end_time

        self.G.add_edge(
            m_node, o_node,
            key=f"assign_{op.operation_template.operation_template_id}_{op.job_instance.job_instance_id}",
            etype="assignment",
            features={
                "start_time":   op.start_time,
                "end_time":     op.end_time,
                "repetition":   op.job_instance.job_instance_id,   # ✔ job_instance_id를 repetition으로 사용
                "label":        f"A({op.job_instance.job_instance_id})",
            }
        )

    # ---------- 호출 지점 ②: OperationInstance.schedule 내부 ----------
    def on_completion(self, op):
        if op.successor is None or op.end_time is None:
            return
        u_node = f"op_{op.operation_template.operation_template_id}"
        v_node = f"op_{op.successor.operation_template.operation_template_id}"

        rep = op.job_instance.job_instance_id  # job_instance_id를 repetition으로 사용
        self.G.add_edge(
            u_node, v_node,
            key=f"compl_{u_node}_{v_node}_{rep}",
            etype="completion",
            features={
                "repetition": rep,             # ✔ 하나만 저장
                "label":      f"C({rep})",
            }
        )