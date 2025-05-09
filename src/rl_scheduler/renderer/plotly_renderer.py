import pandas as pd
import plotly.graph_objs as go
from .renderer import Renderer
from rl_scheduler.scheduler import Scheduler


def _rgba_to_css(rgba) -> str:
    """
    Convert an (r, g, b, a) tuple in 0‑1 range to CSS 'rgba(r,g,b,a)' string
    understood by Plotly.  If the input is already a string, return as‑is.
    """
    if isinstance(rgba, str):
        return rgba
    r, g, b, a = rgba
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})"


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
                        "color": _rgba_to_css(op.job_instance.color),
                        "type_code": op.type_code,
                    }
                )

        if not rows:
            # Nothing scheduled yet → show empty chart but keep machine labels.
            labels = [
                f"Machine {idx}<br>(cap={m.supported_operation_type_codes})"
                for idx, m in enumerate(machine_instances)
            ]

            fig = go.Figure()
            fig.update_layout(
                title=f"{title} (t = {scheduler.timestep})",
                xaxis=dict(title="Time", range=[0, 1]),
                yaxis=dict(
                    title="Machines",
                    tickmode="array",
                    tickvals=list(range(len(labels))),
                    ticktext=labels,
                    autorange="reversed",
                ),
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

        # Iterate over *all* machines so idle ones still appear
        for m_idx, machine in enumerate(machine_instances):
            sub = df[df["machine_id"] == m_idx]
            machine_ability = machine.supported_operation_type_codes
            label = f"Machine {m_idx}<br>(cap={machine_ability})"

            if sub.empty:
                # Add a transparent zero‑length bar so the y‑tick is rendered
                fig.add_trace(
                    go.Bar(
                        x=[0],
                        y=[label],
                        base=[0],
                        orientation="h",
                        marker=dict(color="rgba(0,0,0,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                        opacity=0.0,
                    )
                )
                continue

            customdata = sub[
                ["job_label", "type_code", "start", "duration", "end"]
            ].values

            fig.add_trace(
                go.Bar(
                    x=sub["duration"],
                    y=[label] * len(sub),
                    base=sub["start"],
                    orientation="h",
                    marker=dict(
                        color=sub["color"],
                        line=dict(color="black", width=1),
                    ),
                    customdata=customdata,
                    hovertemplate=(
                        "machine ability: " + str(machine_ability) + "<br>"
                        "job: %{customdata[0]}<br>"
                        "Type: %{customdata[1]}<br>"
                        "Start: %{customdata[2]}<br>"
                        "Duration: %{customdata[3]}<br>"
                        "End: %{customdata[4]}<extra></extra>"
                    ),
                    name=label,
                    showlegend=False,
                )
            )

        # 4) Layout 설정
        if not df.empty:
            min_start = df["start"].min()
            max_end = (df["start"] + df["duration"]).max()
        else:
            min_start, max_end = 0, 1

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
