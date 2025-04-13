import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objs as go

class Renderer:
    # @staticmethod
    # def render_gantt_interactive(machine_instances, render_info_path: Path, title="Interactive Gantt Chart"):
    #     with render_info_path.open("r", encoding="utf-8") as f:
    #         render_info = json.load(f)
    #     color_map = render_info.get("job_colors", {})

    #     rows = []
    #     for m_idx, machine in enumerate(machine_instances):
    #         for op in machine.assigned_operations:
    #             if op.start_time is None or op.end_time is None:
    #                 continue
    #             jt = op.job_instance.job_template.job_template_id
    #             ji = op.job_instance.job_instance_id
    #             rows.append({
    #                 "Machine": f"Machine {m_idx}",
    #                 "TemplateInstance": f"Job{jt}-{ji}",
    #                 "Start": op.start_time,
    #                 "Finish": op.end_time,
    #                 "JobTemplate": str(jt),
    #                 "JobInstance": ji,
    #                 "Type": op.type_code,
    #             })

    #     if not rows:
    #         print("No operations to display.")
    #         return

    #     df = pd.DataFrame(rows)

    #     # 숫자형 축을 위해 Start/Finish를 float형으로 변환
    #     df["Start"] = df["Start"].astype(float)
    #     df["Finish"] = df["Finish"].astype(float)

    #     # Plotly timeline 생성
    #     fig = px.timeline(
    #         df,
    #         x_start="Start",
    #         x_end="Finish",
    #         y="Machine",
    #         color="JobTemplate",                # job_template_id 기준으로 그룹핑
    #         text="TemplateInstance",
    #         hover_data=["JobInstance", "Type"],
    #         title=title,
    #         color_discrete_map=color_map        # ← 여기에 맵을 넘겨줍니다
    #     )

    #     # 테두리 검정색 처리도 bar selector로 통일
    #     fig.update_traces(
    #         marker_line_color="black",
    #         marker_line_width=1,
    #         selector=dict(type="bar")
    #     )
    #     # Y축 위아래 반전(기계 목록이 위에서부터 순서대로 보이도록)
    #     fig.update_yaxes(autorange="reversed")

    #     # X축 범위 및 눈금 설정
    #     max_time = df["Finish"].max()
    #     fig.update_layout(
    #         xaxis=dict(
    #             type="linear",  # 날짜가 아닌 선형 축
    #             range=[0, max_time + 1],
    #             dtick=1        # 1 단위 눈금
    #         ),
    #         margin=dict(l=20, r=250, t=50, b=20)
    #     )

    #     # JobTemplate별로 시리즈를 필터링할 수 있는 드롭다운 버튼 구성
    #     unique_templates = df["JobTemplate"].unique()
    #     buttons = []

    #     # 'All' 버튼
    #     buttons.append(
    #         dict(
    #             label="All",
    #             method="update",
    #             args=[{"visible": [True] * len(fig.data)}]
    #         )
    #     )
    #     # JobTemplate별 버튼
    #     for tpl in unique_templates:
    #         visible = []
    #         for d in fig.data:
    #             # d.name은 color="TemplateInstance" (예: 'Job0-0')
    #             # df에서 해당 TemplateInstance가 tpl과 매칭되는지 확인
    #             series_templates = df[df["TemplateInstance"] == d.name]["JobTemplate"].unique()
    #             visible.append(tpl in series_templates)

    #         buttons.append(
    #             dict(
    #                 label=f"JobTemplate {tpl}",
    #                 method="update",
    #                 args=[{"visible": visible}]
    #             )
    #         )

    #     fig.update_layout(
    #         updatemenus=[
    #             dict(
    #                 type="dropdown",
    #                 showactive=True,
    #                 x=1.15,
    #                 y=1.0,
    #                 xanchor="left",
    #                 yanchor="top",
    #                 buttons=buttons
    #             )
    #         ]
    #     )

    #     # 브라우저에 그래프 표시
    #     fig.show()

    @staticmethod
    def render_gantt_interactive(machine_instances, render_info_path: Path, title="Interactive Gantt Chart"):
        # 1) 색상 설정 불러오기
        render_info = json.loads(render_info_path.read_text(encoding="utf-8"))
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
                    "color_key": jt
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
            fig.add_trace(
                go.Bar(
                    x=sub["duration"],
                    y=[f"Machine {m_idx}"] * len(sub),
                    base=sub["start"],
                    orientation='h',
                    marker=dict(
                        color=[ color_map.get(k, "#cccccc") for k in sub["color_key"] ],
                        line=dict(color="black", width=1)
                    ),
                    hovertemplate=
                        "Task: %{customdata[0]}<br>" +
                        "Start: %{base}<br>" +
                        "End: %{x+base}<extra></extra>",
                    customdata=sub[["job_label"]].values,
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
        legend_items = df[["job_label", "color_key"]].drop_duplicates().sort_values("job_label")
        for _, row in legend_items.iterrows():
            fig.add_trace(
                go.Bar(
                    x=[0], y=[None],
                    marker=dict(color=color_map.get(row["color_key"], "#cccccc")),
                    name=row["job_label"],
                    showlegend=True
                )
            )

        fig.show()

    @staticmethod
    def render_gantt(machine_instances, render_info_path: Path, title="Gantt Chart"):
        with render_info_path.open("r", encoding="utf-8") as f:
            render_info = json.load(f)

        data = []
        for m_idx, machine in enumerate(machine_instances):
            for op in machine.assigned_operations:
                if op.start_time is not None and op.end_time is not None:
                    job_template_id = op.job_instance.job_template.job_template_id
                    job_instance_id = op.job_instance.job_instance_id
                    data.append({
                        "machine_id": m_idx,
                        "job_template_id": job_template_id,
                        "job_instance_id": job_instance_id,
                        "start_time": op.start_time,
                        "end_time": op.end_time,
                        "type_code": op.type_code
                    })

        if not data:
            print("No operations to display.")
            return

        df = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Machines")
        ax.set_yticks(range(len(machine_instances)))
        ax.set_yticklabels([f"Machine {i}" for i in range(len(machine_instances))])

        legend_info = []
        for _, row in df.iterrows():
            jt = row["job_template_id"]
            ji = row["job_instance_id"]
            job_label = f"Job{jt}-{ji}"
            c_map = render_info.get("job_colors", {})
            job_color = c_map.get(str(jt), "#cccccc")

            rect = mpatches.Rectangle(
                (row["start_time"], row["machine_id"] - 0.4),
                row["end_time"] - row["start_time"],
                0.8,
                facecolor=mcolors.to_rgba(job_color),
                edgecolor="black"
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
        plt.show()