# ── rl_scheduler/graph/pad_utils.py ─────────────────────
from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import HeteroData

def heterodata_to_padded(
    data: HeteroData,
    max_mach: int = 32,
    max_op: int   = 128,
    max_e_asg: int = 256,
    max_e_cmp: int = 256,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Convert a (batched) HeteroData into a single 1-D tensor with zero-padding.

    Layout  (same 순서로 concat)          | shape           | dim
    ----------------------------------------------------------------
    machine node feat (id, qlen, busy)    | [max_mach, 3]   |  3
    operation node feat (id, type, dur, …)| [max_op , 4]    |  4
    assignment edges  (src,dst,+3attr)    | [max_e_asg, 5]  |  5
    completion edges  (src,dst,+1attr)    | [max_e_cmp, 3]  |  3
    ----------------------------------------------------------------
    total len = max_mach·3 + max_op·4 + max_e_asg·5 + max_e_cmp·3
    """

    # ---- 1. 노드 ----------------------------------------------------------------
    mach_pad = torch.zeros(max_mach, 3, dtype=torch.float32, device=device)
    op_pad   = torch.zeros(max_op,   4, dtype=torch.float32, device=device)

    n_mach = min(data["machine"].num_nodes, max_mach)
    n_op   = min(data["operation"].num_nodes, max_op)

    mach_pad[:n_mach] = data["machine"].x[:n_mach]
    op_pad[:n_op]     = data["operation"].x[:n_op]

    # ---- 2. assignment edge ------------------------------------------------------
    a_key = ("machine", "assignment", "operation")
    a_edge  = data[a_key].edge_index if a_key in data.edge_index_dict else torch.empty(2,0)
    a_attr  = data[a_key].edge_attr  if hasattr(data[a_key], "edge_attr") else torch.empty(0,3)

    asg_pad = torch.zeros(max_e_asg, 5, dtype=torch.float32, device=device)
    if a_edge.size(1):
        cnt = min(a_edge.size(1), max_e_asg)
        src, dst = a_edge[:, :cnt]
        asg_pad[:cnt, 0] = src
        asg_pad[:cnt, 1] = dst
        asg_pad[:cnt, 2:] = a_attr[:cnt]

    # ---- 3. completion edge ------------------------------------------------------
    c_key = ("operation", "completion", "operation")
    c_edge = data[c_key].edge_index if c_key in data.edge_index_dict else torch.empty(2,0)
    c_attr = data[c_key].edge_attr  if hasattr(data[c_key], "edge_attr") else torch.empty(0,1)

    cmp_pad = torch.zeros(max_e_cmp, 3, dtype=torch.float32, device=device)
    if c_edge.size(1):
        cnt = min(c_edge.size(1), max_e_cmp)
        src, dst = c_edge[:, :cnt]
        cmp_pad[:cnt, 0] = src
        cmp_pad[:cnt, 1] = dst
        cmp_pad[:cnt, 2:] = c_attr[:cnt]

    # ---- 4. flatten --------------------------------------------------------------
    return torch.cat(
        [
            mach_pad.flatten(),
            op_pad.flatten(),
            asg_pad.flatten(),
            cmp_pad.flatten(),
        ],
        dim=0,
    )

def padded_to_heterodata(
    padded: torch.Tensor,
    max_mach: int,
    max_e_asg: int,
    max_e_cmp: int,
    *,
    device: torch.device | None = None,
) -> HeteroData | list[HeteroData]:
    """
    Inverse of `heterodata_to_padded`.

    Parameters
    ----------
    padded      : (B, L) or (L,) tensor produced by `heterodata_to_padded`
    max_mach    : #machines used when padding  (== MAX_N_MACH)
    max_e_asg   : #assignment-edges used when padding
    max_e_cmp   : #completion-edges used when padding
    device      : (optional) device of returned tensors

    Returns
    -------
    HeteroData  (if B == 1)  or  list[HeteroData] (if B > 1)
    """
    if device is None:
        device = padded.device
    if padded.dim() == 1:
        padded = padded.unsqueeze(0)            # (1, L)

    B = padded.size(0)
    MACH_F = 3
    OP_F   = 4
    ASG_F  = 5          # src, dst, 3-attr
    CMP_F  = 3          # src, dst, 1-attr

    max_op = (padded.size(1)
              - max_mach * MACH_F
              - max_e_asg * ASG_F
              - max_e_cmp * CMP_F) // OP_F

    split_sizes = [
        max_mach * MACH_F,
        max_op   * OP_F,
        max_e_asg * ASG_F,
        max_e_cmp * CMP_F,
    ]
    out: list[HeteroData] = []

    for b in range(B):
        mach_blk, op_blk, asg_blk, cmp_blk = torch.split(
            padded[b], split_sizes, dim=0
        )

        # ----- nodes -------------------------------------------------------
        hd = HeteroData()
        mach_raw = mach_blk.view(max_mach, MACH_F)
        op_raw   = op_blk.view(max_op,   OP_F)

        keep_mach = ~(mach_raw == 0).all(dim=1)
        keep_op   = ~(op_raw  == 0).all(dim=1)

        hd["machine"].x   = mach_raw[keep_mach].to(device)
        hd["operation"].x = op_raw [keep_op  ].to(device)

        # index remap (pad-idx → new-idx)
        mach_map = {
            old_idx.item(): new_idx
            for new_idx, old_idx in enumerate(torch.nonzero(keep_mach, as_tuple=False))
        }
        op_map = {
            old_idx.item(): new_idx
            for new_idx, old_idx in enumerate(torch.nonzero(keep_op, as_tuple=False))
        }

        # ----- assignment edges -------------------------------------------
        asg_tbl = asg_blk.view(max_e_asg, ASG_F)
        src, dst, *attr = asg_tbl.split([1, 1, 3], dim=1)
        mask_asg = ~(asg_tbl == 0).all(dim=1)
        src = src[mask_asg].squeeze(1).int().tolist()
        dst = dst[mask_asg].squeeze(1).int().tolist()
        attr = attr[0][mask_asg]

        edge_list, edge_attr = [], []
        for s_raw, d_raw, a in zip(src, dst, attr):
            if s_raw in mach_map and d_raw in op_map:
                edge_list.append([mach_map[s_raw], op_map[d_raw]])
                edge_attr.append(a.tolist())

        if edge_list:
            ei = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
            ea = torch.tensor(edge_attr, dtype=torch.float32, device=device)
            hd["machine", "assignment", "operation"].edge_index = ei
            hd["machine", "assignment", "operation"].edge_attr  = ea

        # ----- completion edges -------------------------------------------
        cmp_tbl = cmp_blk.view(max_e_cmp, CMP_F)
        src, dst, rep = cmp_tbl.split([1, 1, 1], dim=1)
        mask_cmp = ~(cmp_tbl == 0).all(dim=1)
        src = src[mask_cmp].squeeze(1).int().tolist()
        dst = dst[mask_cmp].squeeze(1).int().tolist()
        rep = rep[mask_cmp]

        edge_list, edge_attr = [], []
        for s_raw, d_raw, r in zip(src, dst, rep):
            if s_raw in op_map and d_raw in op_map:
                edge_list.append([op_map[s_raw], op_map[d_raw]])
                edge_attr.append([r.item()])

        if edge_list:
            ei = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
            ea = torch.tensor(edge_attr, dtype=torch.float32, device=device)
            hd["operation", "completion", "operation"].edge_index = ei
            hd["operation", "completion", "operation"].edge_attr  = ea

        out.append(hd)

    return out[0] if B == 1 else out
