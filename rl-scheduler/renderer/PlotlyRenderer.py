import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from .Renderer import Renderer

class PlotlyRenderer(Renderer):
    def __init__(self, scheduler, render_info_path: Path):
        super().__init__(scheduler, render_info_path)

    def render(self, title="Interactive Gantt Chart", mode="browser"):
        machine_instances = self.scheduler.machine_instances
        # 1) 색상 설정 불러오기
        render_info = json.loads(self.render_info_path.read_text(encoding="utf-8"))
        color_map = { str(k): v for k, v in render_info.get("job_colors", {}).items() }

        # 2) DataFrame 준비
        rows = []
        for m_idx, machine in enumerate(machine_instances):
            for op in machine.assigned_operations:
                if op.start_time is None or op.end_time is None:
                    continue
                jt = str(op.job_instance.job_template.job_template_id)
                ji = op.job_instance.job_instance_id
                rows.append({
                    "machine_id": m_idx,
                    "job_label": f"Job{jt}-{ji}",
                    "start": float(op.start_time),
                    "duration": float(op.end_time - op.start_time),
                    "color_key": jt,
                    "template_id": jt,
                    "instance_id": ji,
                    "type_code": op.type_code
                })

        if not rows:
            print("No operations to display.")
            return

        df = pd.DataFrame(rows)

        # 3) Plotly Figure 생성
        fig = go.Figure()

        # machine 별로 trace 추가
        for m_idx in sorted(df["machine_id"].unique()):
            sub = df[df["machine_id"] == m_idx]
            customdata = sub[["job_label", "type_code", "start", "duration"]].values

            fig.add_trace(
                go.Bar(
                    x=sub["duration"],
                    y=[f"Machine {m_idx}"] * len(sub),
                    base=sub["start"],
                    orientation='h',
                    marker=dict(
                        color=[color_map.get(k, "#cccccc") for k in sub["color_key"]],
                        line=dict(color="black", width=1)
                    ),
                    customdata=customdata,
                    hovertemplate=(
                        "Job: %{customdata[0]}<br>"
                        "Type: %{customdata[1]}<br>"
                        "Start: %{customdata[2]}<br>"
                        "Duration: %{customdata[3]}<extra></extra>"
                    ),
                    name=f"Machine {m_idx}",
                    showlegend=False
                )
            )

        # 4) Layout 설정
        min_start = df["start"].min()
        max_end = (df["start"] + df["duration"]).max()

        fig.update_layout(
            title=title,
            barmode='stack',
            xaxis=dict(
                title="Time",
                range=[max(0, min_start - 1), max_end + 1],
                dtick=1
            ),
            margin=dict(l=100, r=50, t=50, b=50),
            yaxis=dict(title="Machines", autorange="reversed")
        )

        # 5) 레전드용 더미 trace 추가 (job_label별)
        legend_items = df[["job_label", "color_key"]].drop_duplicates().sort_values("job_label", ascending=True)
        for _, row in legend_items.iterrows():
            fig.add_trace(
                go.Bar(
                    x=[0], y=[None],
                    marker=dict(color=color_map.get(row["color_key"], "#cccccc")),
                    name=row["job_label"],
                    showlegend=True
                )
            )

        if mode == "streamlit":
            return fig
        elif mode == "browser":
            fig.show()