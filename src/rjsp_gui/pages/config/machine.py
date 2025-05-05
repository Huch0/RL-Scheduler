import json, os
import streamlit as st
from .utils import ensure_state

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 안전장치

# ────────────────────────────────────────────────
#  Machine Template Configuration Page
# ────────────────────────────────────────────────

def render_machine_config() -> None:
    """UI for creating machine templates and exporting to JSON."""
    ensure_state()

    st.header("Machine Configuration")

    # 1️⃣ Add machine template
    st.subheader("Add Machine Template")
    mt_id = st.number_input("Machine ID", min_value=0, step=1, key="machine_template_id")

    # 2️⃣ Supported Op types
    st.subheader("Supported Operation Types")
    supported_ops: list[str] = []
    if st.session_state.type_codes:
        cols = st.columns(len(st.session_state.type_codes))
        for c, code in zip(cols, st.session_state.type_codes):
            with c:
                if st.checkbox(code, key=f"machine_op_{code}"):
                    supported_ops.append(code)
    else:
        st.info("먼저 Operation Type Codes를 설정하세요.")

    # Initialise queue
    if "machines" not in st.session_state:
        st.session_state.machines = []

    if st.button("➕  Add Machine", key="add_machine"):
        st.session_state.machines.append(
            {
                "machine_template_id": int(mt_id),
                "supported_operation_type_codes": supported_ops,
            }
        )
        st.success(f"Machine {mt_id} queued.")

    # 3️⃣ Visualise queue
    if st.session_state.machines:
        st.subheader("Queued Machines")
        st.markdown(
            """
            <style>
            .machine-card{display:flex;justify-content:space-between;align-items:center;
            border:1px solid #d0d0d0;border-radius:8px;padding:0.6rem 1rem;margin-bottom:0.6rem;}
            .remove-btn{background:#ff4d4f;color:#fff;border:none;border-radius:6px;padding:0.4rem 0.6rem;font-weight:700;cursor:pointer;}
            </style>""",
            unsafe_allow_html=True,
        )
        for idx, m in enumerate(st.session_state.machines):
            col1, col2 = st.columns([9, 1])
            with col1:
                st.markdown(
                    f"<div class='machine-card'>"
                    f"<span>Machine {m['machine_template_id']}</span>"
                    f"<span>supports: {', '.join(m['supported_operation_type_codes']) or '—'}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("✖", key=f"rm_{idx}", help="Remove", kwargs={"use_container_width":True}):
                    st.session_state.machines.pop(idx)
                    st.rerun()

        # 4️⃣ Matrix table
        if st.session_state.type_codes:
            st.subheader("Machine × Operation‑Type Matrix")
            header = "| Machine | " + " | ".join(st.session_state.type_codes) + " |\n"
            header += "| --- | " + " | ".join(["---"] * len(st.session_state.type_codes)) + " |\n"
            rows = ""
            for m in st.session_state.machines:
                cells = ["✅" if c in m["supported_operation_type_codes"] else "" for c in st.session_state.type_codes]
                rows += f"| {m['machine_template_id']} | " + " | ".join(cells) + " |\n"
            st.markdown(header + rows)

        # 5️⃣ Export
        # Config name과 version 입력받기
        cfg = st.text_input("Config name", key="machine_cfg")
        ver = st.text_input("Version", "v1", key="machine_ver")
        # JSON 데이터 및 파일명 설정
        json_data = json.dumps({"machines": st.session_state.machines}, indent=4)
        file_name_machines = f"M-{cfg or 'default'}-{ver}.json"
        st.download_button(
            label="Export JSON",
            data=json_data,
            file_name=file_name_machines,
            mime="application/json",
        )
