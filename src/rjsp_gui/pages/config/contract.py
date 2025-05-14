"""Contract Configuration Tab – Plotly (deterministic preview)
"""
from __future__ import annotations

import json
from typing import List

import streamlit as st

from rjsp_gui.pages.config.utils import ensure_state
from rjsp_gui.pages.config.job import _timeline_html
from rjsp_gui.services.plotly_service import plot_profit_curves

# ---------------------------------------------------------------------
# 🖥️  Main UI
# ---------------------------------------------------------------------

def render_contract_config() -> None:
    """Streamlit tab for contract editing with deterministic Plotly preview."""
    ensure_state()

    # Optional: load existing contracts
    contracts_file = st.file_uploader("Load contracts JSON", type=["json"], key="load_contracts")
    if contracts_file:
        try:
            raw   = contracts_file.read().decode("utf-8")
            clean = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("//"))
            st.session_state.contracts = json.loads(clean).get("contracts", {})
            st.success("Contracts configuration loaded.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load contracts JSON: {exc}")

    st.header("Contract Configuration")

    if not st.session_state.job_templates:
        st.info("먼저 Job Template을 생성하세요.")
        return

    st.session_state.setdefault("contracts", {})

    st.subheader("Job‑wise Contracts")

    for jt in st.session_state.job_templates:
        job_key  = f"job_{jt['job_template_id']}"
        existing = st.session_state.contracts.get(job_key, [])
        color    = jt.get("color", "#aaa")

        with st.expander(f"Job Template {jt['job_template_id']}"):
            if existing:
                st.caption(f"Loaded {len(existing)} instances for {job_key}")

            # Timeline ------------------------------------------------------
            ops = [
                next(o for o in st.session_state.operation_templates if o["operation_template_id"] == oid)
                for oid in jt["operation_template_sequence"]
            ]
            st.markdown(_timeline_html(ops, color), unsafe_allow_html=True)

            # Repetition ----------------------------------------------------
            default_rep = len(existing) or 1
            repetitions = st.number_input("Repetition", 1, step=1, value=default_rep, key=f"rep_{job_key}")

            # Table header --------------------------------------------------
            hdr = st.columns(4)
            hdr[0].markdown("**Idx**")
            hdr[1].markdown("**Price**")
            hdr[2].markdown("**Deadline**")
            hdr[3].markdown("**Late penalty**")

            prices: List[float] = []
            dls:    List[int]   = []
            lps:    List[float] = []

            for i in range(repetitions):
                c0, c1, c2, c3 = st.columns(4)
                c0.markdown(str(i))

                # defaults --------------------------------------------------
                price_def = existing[i]["price"]        if i < len(existing) else 1000.0
                dl_def    = existing[i]["deadline"]     if i < len(existing) else 1
                lp_def    = existing[i]["late_penalty"] if i < len(existing) else 0.0

                prices.append(c1.number_input("price",   0.0, value=price_def, key=f"p_{job_key}_{i}"))
                dls.append(c2.number_input("deadline", 1, value=dl_def, key=f"d_{job_key}_{i}"))
                lps.append(c3.number_input("penalty", 0.0, value=lp_def, key=f"lp_{job_key}_{i}"))

            # Plotly preview -----------------------------------------------
            if prices:
                st.caption("Profit‑curve preview (deterministic)")
                plot_profit_curves(prices, dls, lps, key=f"plot_{job_key}")

            # Save ----------------------------------------------------------
            if st.button("Save instances", key=f"save_{job_key}"):
                st.session_state.contracts[job_key] = [
                    {
                        "job_instance_id": i,
                        "price": prices[i],
                        "deadline": dls[i],
                        "late_penalty": lps[i],
                    }
                    for i in range(repetitions)
                ]
                st.success(f"Contract for {job_key} saved.")

    # Export ---------------------------------------------------------------
    if st.session_state.contracts:
        st.divider()
        cfg = st.text_input("Config name", key="contract_cfg")
        ver = st.text_input("Version", "v1", key="contract_ver")
        file_name = f"C-{cfg or 'default'}-{ver}.json"
        st.download_button(
            "Download contracts JSON",
            json.dumps({"contracts": st.session_state.contracts}, indent=4),
            file_name=file_name,
            mime="application/json",
        )