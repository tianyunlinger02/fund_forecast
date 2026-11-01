"""Small, local visual layer shared by the everyday fund pages."""
from pathlib import Path

import streamlit as st


def apply_theme() -> str:
    # Native theme toggles do not rerun Python. This snapshot is for charts only;
    # page styles must follow Streamlit's live colors independently of it.
    theme = "dark" if st.context.theme.type == "dark" else "light"
    css = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    # Only developer-owned CSS enters HTML; fund/provider text uses native widgets.
    st.html(f"<style>{css}</style>")
    return theme
