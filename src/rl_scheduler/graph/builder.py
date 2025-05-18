# ── builder.py ────────────────────────────────────────────
import networkx as nx

# ---------- 1. NetworkX 그래프 생성 ----------
def build_graph_from_scheduler(sch) -> nx.MultiDiGraph:
    """
    Convert a Scheduler instance into a directed multigraph.

    Node ids:
        op_{instance_id}
        mach_{machine_id}
    """
    G = nx.MultiDiGraph()

    # 1-A. Machine nodes
    for m in sch.machine_instances:
        mt_id = m.machine_template.machine_template_id
        G.add_node(
            f"mach_{mt_id}", ntype="machine",
            features={
                "id": mt_id,
                "supported_types": m.machine_template.supported_operation_type_codes,
                "queue_len": len(getattr(m, "assigned_operations", [])),
                "busy_until": getattr(m, "last_assigned_end_time", 0.0),
            },
        )

    # 1-B. Operation nodes
    for job_type in sch.job_instances:
        for job_inst in job_type:
            for op in job_inst.operation_instance_sequence:
                ot_id = op.operation_template.operation_template_id
                G.add_node(
                    f"op_{ot_id}", ntype="operation",
                    features={
                        "id": ot_id,
                        "type": op.type_code,
                        "duration": op.duration,
                        "job_id": job_inst.job_template.job_template_id,
                    },
                )

                # logical edge for precedence
                if op.successor:
                    succ_ot_id = op.successor.operation_template.operation_template_id
                    G.add_edge(
                        f"op_{ot_id}", f"op_{succ_ot_id}",
                        key=f"log_{ot_id}_{succ_ot_id}", etype="logical",
                        features={"label": "L"},
                    )
                    G.add_edge(
                        f"op_{succ_ot_id}", f"op_{ot_id}",
                        key=f"log_{succ_ot_id}_{ot_id}", etype="logical",
                        features={"label": "L"},
                    )

    # 1-C. Eligibility edges (type_valid)
    for m in sch.machine_instances:
        m_id = m.machine_template.machine_template_id
        for n, d in G.nodes(data=True):
            if d.get("ntype") != "operation":
                continue
            if d["features"]["type"] in m.machine_template.supported_operation_type_codes:
                G.add_edge(
                    f"mach_{m_id}", n,
                    key=f"type_valid_{m_id}_{n}", etype="type_valid",
                    features={"label": "T"},
                )

    # 1-D. Dynamic assignment and completion edges
    for job_type in sch.job_instances:
        for job_inst in job_type:
            for op in job_inst.operation_instance_sequence:
                # assignment edge
                if op.processing_machine:
                    mt_id = op.processing_machine.machine_template.machine_template_id
                    ot_id = op.operation_template.operation_template_id
                    G.add_edge(
                        f"mach_{mt_id}", f"op_{ot_id}",
                        key=f"assign_{ot_id}", etype="assignment",
                        features={
                            "start_time": op.start_time,
                            "end_time": op.end_time,
                            "job_instance_id": job_inst.job_instance_id,
                            "label": f"A({job_inst.job_instance_id})",
                        },
                    )
                # completion edge
                if op.end_time is not None and op.successor:
                    ot_id = op.operation_template.operation_template_id
                    succ_ot_id = op.successor.operation_template.operation_template_id
                    G.add_edge(
                        f"op_{ot_id}", f"op_{succ_ot_id}",
                        key=f"compl_{ot_id}", etype="completion",
                        features={
                            "completion_time": op.end_time,
                            "job_instance_id": job_inst.job_instance_id,
                            "label": f"C({job_inst.job_instance_id})",
                        },
                    )
    return G
