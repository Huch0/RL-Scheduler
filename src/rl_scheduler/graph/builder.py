# ── builder.py (patched) ──────────────────────────────────
import networkx as nx
from typing import Dict, Set, Union, Sequence

def _safe_busy_until(last_end: Union[int, float, Sequence[int]]) -> float:
    """Return scalar busy_until regardless of scalar/list input."""
    if isinstance(last_end, (list, tuple)):
        return max(last_end) if last_end else 0.0
    return float(last_end) if last_end is not None else 0.0

def build_graph_from_scheduler(sch) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    # ---------- type-code encoding ----------
    type_code_set: Set[str] = set()
    for job_type in sch.job_instances:
        for job_inst in job_type:
            for op in job_inst.operation_instance_sequence:
                type_code_set.add(op.type_code)
    for m in sch.machine_instances:
        type_code_set.update(m.machine_template.supported_operation_type_codes)

    type_code_to_int: Dict[str, int] = {
        code: idx for idx, code in enumerate(sorted(type_code_set))
    }

    # ---------- 1-A. machine nodes ----------
    for m in sch.machine_instances:
        mt_id = m.machine_template.machine_template_id
        busy_raw = getattr(m, "last_assigned_end_time", 0)
        G.add_node(
            f"mach_{mt_id}",
            ntype="machine",
            features={
                "id": mt_id,
                "queue_len": len(getattr(m, "assigned_operations", [])),
                "busy_until": _safe_busy_until(busy_raw),
            },
        )

    # ---------- 1-B. operation nodes + logical edges ----------
    for job_type in sch.job_instances:
        for job_inst in job_type:
            for op in job_inst.operation_instance_sequence:
                ot_id = op.operation_template.operation_template_id
                G.add_node(
                    f"op_{ot_id}",
                    ntype="operation",
                    features={
                        "id": ot_id,
                        "type": type_code_to_int[op.type_code],
                        "duration": op.duration,
                        "job_id": job_inst.job_template.job_template_id,
                    },
                )
                if op.successor:
                    succ_ot = op.successor.operation_template.operation_template_id
                    for u, v in ((ot_id, succ_ot), (succ_ot, ot_id)):
                        G.add_edge(
                            f"op_{u}",
                            f"op_{v}",
                            key=f"log_{u}_{v}",
                            etype="logical",
                            features={"label": "L"},
                        )

    # ---------- 1-C. type-valid edges (op → mach) ----------
    for m in sch.machine_instances:
        m_id = m.machine_template.machine_template_id
        supported = {type_code_to_int[c]
                     for c in m.machine_template.supported_operation_type_codes}
        for n, data in G.nodes(data=True):
            if data["ntype"] != "operation":
                continue
            if data["features"]["type"] in supported:
                G.add_edge(
                    n,
                    f"mach_{m_id}",
                    key=f"type_valid_{n}_{m_id}",
                    etype="type_valid",
                    features={"label": "T"},
                )

    # ---------- 1-D. assignment / completion edges ----------
    for job_type in sch.job_instances:
        for job_inst in job_type:
            rep = job_inst.job_instance_id  # job_instance_id를 repetition으로 사용
            for op in job_inst.operation_instance_sequence:
                ot_id = op.operation_template.operation_template_id

                # assignment (m → o)
                if op.processing_machine:
                    mt_id = op.processing_machine.machine_template.machine_template_id
                    G.add_edge(
                        f"mach_{mt_id}",
                        f"op_{ot_id}",
                        key=f"assign_{mt_id}_{ot_id}_{rep}",
                        etype="assignment",
                        features={
                            "start_time": op.start_time,
                            "end_time": op.end_time,
                            "repetition": rep,
                            "label": f"A({rep})",
                        },
                    )

                # completion (o → succ-o)
                if op.end_time is not None and op.successor:
                    succ_ot = op.successor.operation_template.operation_template_id
                    G.add_edge(
                        f"op_{ot_id}",
                        f"op_{succ_ot}",
                        key=f"compl_{ot_id}_{succ_ot}_{rep}",
                        etype="completion",
                        features={
                            "repetition": rep,        # spec-compliant
                            "label": f"C({rep})",
                        },
                    )

    # store mapping
    G.graph["type_code_encoding"] = type_code_to_int
    G.graph["int_to_type_code"] = {v: k for k, v in type_code_to_int.items()}
    return G
