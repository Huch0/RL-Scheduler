import json
import streamlit as st
from .utils import ensure_state, tag, shade

# ────────────────────────────────────────────────
# 🔨  Timeline visualisation helpers
# ────────────────────────────────────────────────

def _timeline_html(ops, colour: str) -> str:
    """Return a Gantt‑style HTML timeline (diagonal layout).

    • Bars are proportional to duration.
    • Label above each bar: "type: X / duration: Y".
    • **Only *one* end‑time marker** (overall finish) is shown at the bottom‑right.
    """
    if not ops:
        return ""

    total = sum(o["duration"] for o in ops)
    html_rows = []
    cum_start = 0

    for idx, op in enumerate(ops):
        start  = cum_start
        end    = cum_start + op["duration"]
        left   = start / total * 100  # %
        width  = op["duration"] / total * 100  # %
        mid    = left + width / 2
        cum_start = end

        bar_colour = shade(colour, 0.08 * idx)

        label = (
            f"<span style='position:absolute;top:-14px;left:{mid}%;transform:translateX(-50%);"
            "font-size:0.75rem;font-weight:500;'>"
            f"type: {op['type_code']} / duration: {op['duration']}"
            "</span>"
        )

        html_rows.append(
            f"<div style='position:relative;height:32px;margin-top:6px;'>"
            f"  <div style='background:{bar_colour};height:20px;width:{width}%;margin-left:{left}%;"
            "border-radius:6px;'></div>"
            f"  {label}"
            "</div>"
        )

    # single global end‑time marker (bottom‑right)
    end_marker = (
        f"<div style='text-align:right;font-size:0.75rem;margin-top:4px;color:#444;'>"
        f"{total}</div>"
    )

    return (
        "<div style='border:1px solid #ddd;border-radius:8px;padding:18px 12px 12px;overflow-x:auto;'>"
        + "".join(html_rows)
        + end_marker
        + "</div>"
    )

# ────────────────────────────────────────────────
# 🧩  Main UI
# ────────────────────────────────────────────────

def render_job_config() -> None:
    ensure_state()

    # ────────────────────────────────────────────────
    # 🔄  Load existing configurations
    ops_file = st.file_uploader("Load operations JSON", type=["json"], key="load_ops")
    if ops_file:
        try:
            raw = ops_file.read().decode('utf-8')
            clean = '\n'.join(line for line in raw.splitlines() if not line.strip().startswith('//'))
            ops_data = json.loads(clean)
            st.session_state.operation_templates = ops_data.get("operations", [])
            # derive type codes from operations
            st.session_state.type_codes = sorted({op.get("type_code") for op in st.session_state.operation_templates if op.get("type_code")})
            st.success("Operations configuration loaded.")
            st.write(f"Loaded {len(st.session_state.operation_templates)} operations.")
        except Exception as e:
            st.error(f"Failed to load operations JSON: {e}")
    jobs_file = st.file_uploader("Load jobs JSON", type=["json"], key="load_jobs")
    if jobs_file:
        try:
            raw = jobs_file.read().decode('utf-8')
            clean = '\n'.join(line for line in raw.splitlines() if not line.strip().startswith('//'))
            jobs_data = json.loads(clean)
            st.session_state.job_templates = jobs_data.get("jobs", [])
            st.success("Jobs configuration loaded.")
            st.write(f"Loaded {len(st.session_state.job_templates)} jobs.")
        except Exception as e:
            st.error(f"Failed to load jobs JSON: {e}")

    st.header("Job Configuration")

    # 0️⃣  Operation Type Codes
    st.subheader("Operation Type Codes")
    with st.form("type_codes_form", clear_on_submit=True):
        new_code = st.text_input("Add new type code", placeholder="e.g. Drill")
        if st.form_submit_button("➕  Add"):
            if not new_code:
                st.warning("입력값이 비어 있습니다.")
            elif new_code in st.session_state.type_codes:
                st.warning("이미 존재하는 코드입니다.")
            else:
                st.session_state.type_codes.append(new_code)
                st.success(f"'{new_code}' 추가")

    # 목록 & 삭제
    if st.session_state.type_codes:
        cols = st.columns(len(st.session_state.type_codes))
        for code, col in zip(st.session_state.type_codes, cols):
            with col:
                st.markdown(tag(code), unsafe_allow_html=True)
                if st.button("🗑", key=f"del_{code}"):
                    used = any(op["type_code"] == code for op in st.session_state.operation_templates)
                    if used:
                        st.error("사용 중인 코드입니다.")
                    else:
                        st.session_state.type_codes.remove(code)
                        st.rerun()
    else:
        st.info("먼저 Type 코드를 추가하세요.")

    st.divider()

    # 1️⃣  Operation Template Palette
    st.subheader("Operation Template Palette")
    if st.session_state.type_codes:
        with st.form("add_op_form", clear_on_submit=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                op_id_str = st.text_input("Operation ID (required)")
            with col2:
                type_code = st.selectbox("Type Code", st.session_state.type_codes)
            with col3:
                duration = st.number_input("Duration", min_value=1, value=1)
            with col4:
                job_id_str = st.text_input("Job Template ID (required)")
            if st.form_submit_button("Add Operation"):
                if not (op_id_str.isdigit() and job_id_str.isdigit()):
                    st.warning("ID 값은 숫자여야 합니다.")
                elif any(op["operation_template_id"] == int(op_id_str) for op in st.session_state.operation_templates):
                    st.warning("이미 존재하는 Operation ID입니다.")
                else:
                    st.session_state.operation_templates.append(
                        {
                            "operation_template_id": int(op_id_str),
                            "type_code": type_code,
                            "duration": duration,
                            "job_template_id": int(job_id_str),
                        }
                    )
                    st.success(f"Operation {op_id_str} 추가")

    # 팔레트 표시
    for idx, op in enumerate(sorted(st.session_state.operation_templates, key=lambda o: o["operation_template_id"])):
        colA, colB = st.columns([6, 1])
        with colA:
            st.markdown(tag(f"{op['operation_template_id']}:{op['type_code']} ({op['duration']})"), unsafe_allow_html=True)
        with colB:
            if st.button("❌", key=f"op_rm_{op['operation_template_id']}"):
                st.session_state.operation_templates.pop(idx)
                st.rerun()

    # 2️⃣  Job Template Builder
    st.divider()
    st.subheader("Build Job Template")

    job_id_input = st.text_input("Job Template ID (required)")
    job_color = st.color_picker("Job Color", "#ADB5BD")

    # Preview timeline
    if job_id_input.isdigit():
        tmp_ops = [op for op in st.session_state.operation_templates if op["job_template_id"] == int(job_id_input)]
        if tmp_ops:
            tmp_ops.sort(key=lambda o: o["operation_template_id"])
            st.markdown(_timeline_html(tmp_ops, job_color), unsafe_allow_html=True)

    if st.button("Add Job Template"):
        if not job_id_input.isdigit():
            st.warning("Job Template ID는 숫자 필수입니다.")
        elif any(jt["job_template_id"] == int(job_id_input) for jt in st.session_state.job_templates):
            st.warning("중복된 Job Template ID입니다.")
        else:
            job_id = int(job_id_input)
            ops_for_job = [op for op in st.session_state.operation_templates if op["job_template_id"] == job_id]
            if not ops_for_job:
                st.warning("해당 Job ID에 연결된 Operation이 없습니다.")
            else:
                ops_for_job.sort(key=lambda o: o["operation_template_id"])
                op_ids = [o["operation_template_id"] for o in ops_for_job]

                st.session_state.job_templates.append(
                    {
                        "job_template_id": job_id,
                        "operation_template_sequence": op_ids,
                        "color": job_color,
                    }
                )
                st.success(f"Job {job_id} 저장")
                st.rerun()

    # 3️⃣  Queue
    if st.session_state.job_templates:
        st.divider()
        st.subheader("Job Template Queue")
        for jt in st.session_state.job_templates:
            ops = [next(o for o in st.session_state.operation_templates if o["operation_template_id"] == oid)
                   for oid in jt["operation_template_sequence"]]
            ops.sort(key=lambda o: o["operation_template_id"])
            st.markdown(tag(f"Job {jt['job_template_id']}", jt["color"]), unsafe_allow_html=True)
            st.markdown(_timeline_html(ops, jt["color"]), unsafe_allow_html=True)

    # 4️⃣  Export
    st.divider()
    cfg = st.text_input("Config name", key="job_cfg")
    ver = st.text_input("Version", "v1", key="job_ver")

    # jobs_export 리스트를 명시적으로 만들어 color까지 포함
    jobs_export = [
        {
            "job_template_id": jt["job_template_id"],
            "operation_template_sequence": jt["operation_template_sequence"],
            "color": jt["color"],
        }
        for jt in st.session_state.job_templates
    ]
    jobs_json = json.dumps({"jobs": jobs_export}, indent=4)

    st.download_button(
        "Download jobs JSON",
        jobs_json,
        file_name=f"J-{cfg or 'default'}-0-{len(jobs_export)-1}.json",
        mime="application/json",
        disabled=not jobs_export,
    )

    ops_json = json.dumps({"operations": st.session_state.operation_templates}, indent=4)

    st.download_button(
        "Download operations JSON",
        ops_json,
        file_name=f"O-{cfg}-{ver}.json" if cfg else "operations.json",
        disabled=not st.session_state.operation_templates,
    )
