from __future__ import annotations
from matplotlib import pyplot as plt

from .job_template import JobTemplate
from ..operation.operation_instance import OperationInstance
from ..profit import ProfitFunction
from typing import List, Tuple

RGBA = Tuple[float, float, float, float]  # (R, G, B, A) 0‑1


class JobInstance:
    def __init__(
        self,
        job_instance_id: int,
        job_template: JobTemplate,
        color: RGBA,
        profit_fn: ProfitFunction,
    ):
        self.job_instance_id = job_instance_id
        self.job_template = job_template

        self.color = color

        self.profit_fn = profit_fn
        self.operation_instance_sequence = None
        self.next_op_idx = 0  # Tracks the next operation to be scheduled
        self.completed = False  # Tracks whether the job is completed

    def set_operation_instance_sequence(
        self, operation_instance_sequence: List[OperationInstance]
    ):
        self.operation_instance_sequence = operation_instance_sequence
        for operation_instance in self.operation_instance_sequence:
            operation_instance.set_job_instance(self)

    def __str__(self):
        template_id = getattr(self.job_template, "job_template_id", "N/A")
        profit = f"price={self.profit_fn.price}" if self.profit_fn else "NoProfit"
        ops = (
            len(self.operation_instance_sequence)
            if self.operation_instance_sequence
            else 0
        )
        return f"""JobInstance(id={self.job_instance_id}, template_id=
        {template_id}, {profit}, ops_count={ops})"""

    def plot(self):
        """
        Create a Matplotlib Figure that shows:
        1. Operation sequence as horizontal bars (top subplot)
        2. ProfitFunction curve (bottom subplot)
        Both share the same time axis so they sit inside one rectangle.
        """
        ops = self.operation_instance_sequence
        prof = getattr(self, "profit_fn", None)

        # compute start/end times for operations (serial schedule)
        starts, widths = [], []
        current = 0
        for op in ops:
            starts.append(current)
            widths.append(op.duration)
            current += op.duration
        total_time = max(current, getattr(prof, "deadline", 10) * 2)

        fig, (ax_ops, ax_profit) = plt.subplots(
            2, 1, figsize=(5, 2.5), sharex=True, height_ratios=[2, 1]
        )
        # --- operations ---
        for idx, (s, w, op) in enumerate(zip(starts, widths, ops)):
            # Draw differently if the operation has been scheduled (end_time set)
            if op.end_time is not None:
                # hatched, transparent fill with coloured edge
                ax_ops.broken_barh(
                    [(s, w)],
                    (idx, 0.8),
                    facecolors="none",
                    edgecolor=self.color,
                    hatch="///",
                    linewidth=1.5,
                )
            else:
                # unscheduled: solid fill
                ax_ops.broken_barh(
                    [(s, w)],
                    (idx, 0.8),
                    facecolors=self.color,
                )

            # label operation type and duration at the center of the bar
            label = f"{op.type_code}\n{op.duration}"
            ax_ops.text(
                s + w / 2,
                idx + 0.4,
                label,
                ha="center",
                va="center",
                color="black",
                fontsize=7,
            )
        ax_ops.set_yticks([])
        ax_ops.set_xlim(left=0, right=total_time)
        ax_ops.set_xlabel("")  # hide
        ax_ops.set_ylabel("Ops")
        ax_ops.grid(True, axis="x", linestyle="--", alpha=0.3)

        # --- profit ---
        # Delegate plotting to ProfitFunction itself, drawing on ax_profit
        prof.plot(ax=ax_profit)

        ax_profit.set_xlim(left=0, right=total_time)
        ax_profit.set_ylim(bottom=0)
        ax_profit.set_xlabel("Time")
        ax_profit.set_ylabel("Profit")
        ax_profit.grid(True, linestyle="--", alpha=0.3)

        fig.tight_layout()
        return fig

    def get_profit(self) -> float:
        """
        Calculate the profit at a given time.
        If the job is completed, return the profit function value.
        """
        if self.completed:
            return self.profit_fn(self.operation_instance_sequence[-1].end_time)
        else:
            return 0.0
