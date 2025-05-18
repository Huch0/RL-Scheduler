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

        # (2) Eligibility edge(기본) 제거 → Assignment edge 추가
        elig_key = f"elig_{machine.machine_template.machine_template_id}_{o_node}"
        if self.G.has_edge(m_node, o_node, key=elig_key):
            self.G.remove_edge(m_node, o_node, key=elig_key)

        self.G.add_edge(
            m_node, o_node,
            key=f"assign_{op.operation_template.operation_template_id}",
            etype="assignment",
            features={
                "start_time": op.start_time,
                "end_time":   op.end_time,
                "job_instance_id": op.job_instance.job_instance_id
            }
        )

    # ---------- 호출 지점 ②: OperationInstance.schedule 내부 ----------
    def on_completion(self, op):
        """Update graph when an operation finishes (end_time set)."""
        if op.successor is None or op.end_time is None:
            return  # nothing to do

        u_node = f"op_{op.operation_template.operation_template_id}"
        v_node = f"op_{op.successor.operation_template.operation_template_id}"

        self.G.add_edge(
            u_node, v_node,
            key=f"compl_{op.operation_template.operation_template_id}_{datetime.utcnow().timestamp()}",
            etype="completion",
            features={
                "completion_time": op.end_time,
                "job_instance_id": op.job_instance.job_instance_id
            }
        )
