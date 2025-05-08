import pandas as pd
import plotly.graph_objs as go
from .renderer import Renderer
from rl_scheduler.scheduler import Scheduler


class PlotlyRenderer(Renderer):
    @staticmethod
    def render(
        scheduler: Scheduler, title: str = "Machine Schedule"
    ) -> go.Figure | None:
        """
        Build a Plotly Gantt‑style figure of the current scheduler state.

        Parameters
        ----------
        scheduler : Scheduler
            The scheduler whose machine & job status will be visualised.
        title : str
            Figure title.

        Returns
        -------
        plotly.graph_objs.Figure | None
            A Plotly figure ready for `st.plotly_chart`, or *None* if there are
            no scheduled operations yet.
        """
        machine_instances = scheduler.machine_instances

        # DataFrame 준비
        rows = []
        for m_idx, machine in enumerate(machine_instances):
            for op in machine.assigned_operations:
                if op.start_time is None or op.end_time is None:
                    continue
                jt = str(op.job_instance.job_template.job_template_id)
                ji = op.job_instance.job_instance_id
                rows.append(
                    {
                        "machine_id": m_idx,
                        "job_label": f"job{jt}-{ji}",
                        "start": float(op.start_time),
                        "duration": float(op.end_time - op.start_time),
                        "end": float(op.end_time),
                        "color": op.job_instance.color,  # RGBA
                        "type_code": op.type_code,
                    }
                )

        if not rows:
            # Nothing scheduled yet → return an empty figure with a placeholder.
            fig = go.Figure()
            fig.update_layout(
                title=f"{title} (t = {scheduler.timestep})",
                xaxis=dict(title="Time", range=[0, 1]),
                yaxis=dict(title="Machines", visible=False),
                annotations=[
                    dict(
                        text="No scheduled operations",
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=16, color="gray"),
                        x=0.5,
                        y=0.5,
                    )
                ],
                margin=dict(l=100, r=50, t=50, b=50),
            )
            return fig

        df = pd.DataFrame(rows)

        # 3) Plotly Figure 생성
        fig = go.Figure()

        # machine 별로 trace 추가
        for m_idx in sorted(df["machine_id"].unique()):
            sub = df[df["machine_id"] == m_idx]
            customdata = sub[
                ["job_label", "type_code", "start", "duration", "end"]
            ].values

            fig.add_trace(
                go.Bar(
                    x=sub["duration"],
                    y=[f"Machine {m_idx}"] * len(sub),
                    base=sub["start"],
                    orientation="h",
                    marker=dict(
                        color=sub["color"],
                        line=dict(color="black", width=1),
                    ),
                    customdata=customdata,
                    hovertemplate=(
                        "job: %{customdata[0]}<br>"
                        "Type: %{customdata[1]}<br>"
                        "Start: %{customdata[2]}<br>"
                        "Duration: %{customdata[3]}<br>"
                        "End: %{customdata[4]}<extra></extra>"
                    ),
                    name=f"Machine {m_idx}",
                    showlegend=False,
                )
            )

        # 4) Layout 설정
        min_start = df["start"].min()
        max_end = (df["start"] + df["duration"]).max()

        fig.update_layout(
            title=title,
            barmode="stack",
            xaxis=dict(
                title="Time", range=[max(0, min_start - 1), max_end + 1], dtick=1
            ),
            margin=dict(l=100, r=50, t=50, b=50),
            yaxis=dict(title="machines", autorange="reversed"),
        )

        return fig
