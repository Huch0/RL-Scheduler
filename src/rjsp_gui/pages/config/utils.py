import streamlit as st

# ───────────────────────────────────────────────
# 📦  Session‑state helpers
# ───────────────────────────────────────────────
DEFAULT_STATE = {
    "machine_templates": [],
    "next_machine_id": 0,
    "type_codes": [],
    "operation_templates": [],
    "job_templates": [],
}

def ensure_state() -> None:
    """Ensure default keys are present in st.session_state."""
    for k, v in DEFAULT_STATE.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ───────────────────────────────────────────────
# 🎨  Color helpers
# ───────────────────────────────────────────────
def _hex2rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

def _rgb2hex(rgb):
    return "#" + "".join(f"{c:02X}" for c in rgb)

def shade(hex_color: str, factor: float = 0.15) -> str:
    """Lighten <hex_color> by <factor> (0‑1)."""
    r, g, b = _hex2rgb(hex_color)
    r = min(int(r + (255 - r) * factor), 255)
    g = min(int(g + (255 - g) * factor), 255)
    b = min(int(b + (255 - b) * factor), 255)
    return _rgb2hex((r, g, b))

# ───────────────────────────────────────────────
# 🏷️  HTML tag helper
# ───────────────────────────────────────────────
def tag(label: str, color: str = "#E9ECEF") -> str:
    return (
        f"<span style='background:{color};padding:4px 8px;border-radius:6px;"
        f"font-size:0.8rem;margin:2px;display:inline-block;'>{label}</span>"
    )
