import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from .renderer import Renderer


class MatplotRenderer(Renderer):
    @staticmethod
    def render(scheduler, title: str | None = None):
        if title is None:
            title = f"Machine Schedule (t = {scheduler.timestep})"

        machine_instances = scheduler.machine_instances

        data = []
        for m_idx, machine in enumerate(machine_instances):
            for op in machine.assigned_operations:
                if op.start_time is not None and op.end_time is not None:
                    job_template_id = op.job_instance.job_template.job_template_id
                    job_instance_id = op.job_instance.job_instance_id
                    data.append(
                        {
                            "machine_id": m_idx,
                            "job_template_id": job_template_id,
                            "job_instance_id": job_instance_id,
                            "start_time": op.start_time,
                            "end_time": op.end_time,
                            "type_code": op.type_code,
                        }
                    )

        if not data:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.set_title(title)
            ax.text(0.5, 0.5, "No scheduled operations", ha="center", va="center")
            ax.axis("off")
            return fig

        df = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("machines")
        ax.set_yticks(range(len(machine_instances)))
        ax.set_yticklabels([f"Machine {i}" for i in range(len(machine_instances))])

        legend_info = []
        for _, row in df.iterrows():
            jt = row["job_template_id"]
            ji = row["job_instance_id"]
            job_label = f"Job{jt}-{ji}"
            job_color = op.job_instance.color

            rect = mpatches.Rectangle(
                (row["start_time"], row["machine_id"] - 0.4),
                row["end_time"] - row["start_time"],
                0.8,
                facecolor=mcolors.to_rgba(job_color),
                edgecolor="black",
            )
            ax.add_patch(rect)
            if job_label not in [x[2] for x in legend_info]:
                legend_info.append((jt, ji, job_label, job_color))

        legend_info.sort(key=lambda x: (x[0], x[1]))
        legend_patches = [
            mpatches.Patch(color=mcolors.to_rgba(item[3]), label=item[2])
            for item in legend_info
        ]
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left")

        min_time = df["start_time"].min()
        max_time = df["end_time"].max()
        ax.set_xlim(left=max(min_time - 1, 0), right=max_time + 1)
        ax.set_ylim(-1, len(machine_instances))

        plt.tight_layout()
        return fig
