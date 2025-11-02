
"""
@Project Name: Knowledge_safety

@File Name: app.py
@Software: Python
@Time: Sep/2025
@Author: Peng Su and Rui Xu
@Contact: pengsu94@outlook.com rxu@kth.se
@Version: 0.2.7
"""
import os
import runpy
import types
import streamlit as st
from pathlib import Path

# ---- IMPORTANT: make page config here so downstream apps don't error ----
st.set_page_config(page_title="Integrated App", layout="wide")

# Paths to the two apps (adjust if you move the files)
APP_PATHS = {
    "Neo4j RAG Q&A": Path("/Users/pengsu_workstation/Publication/LLM_Requirement_Analysis/Knowledge_Graph_RAG/LLM_app_v0.2.py"),
    "Prolog Rules Tool": Path("/Users/pengsu_workstation/Publication/LLM_Requirement_Analysis/Knowledge_Graph_RAG/test_app_fixed.py"),
}

# Sidebar navigation
with st.sidebar:
    st.header("📚 Apps")
    choice = st.radio("Choose a tool", list(APP_PATHS.keys()), index=0)

# Helper: patch set_page_config in sub-apps to a no-op so our top-level config is the only one used.
def _patch_streamlit_config_noop():
    # Replace st.set_page_config with a no-op for the duration of sub-app execution
    st.set_page_config = lambda *args, **kwargs: None  # type: ignore

# Helper: isolate execution of a sub-app file
def run_sub_app(file_path: Path):
    if not file_path.exists():
        st.error(f"File not found: {file_path}")
        return
    _patch_streamlit_config_noop()
    # Provide an isolated global namespace so sub-app variables don't leak
    # but still run as if __main__ so Streamlit code executes normally.
    runpy.run_path(str(file_path), run_name="__main__")

st.markdown(f"### 🔀 Integrated App → **{choice}**")

selected_path = APP_PATHS[choice]
run_sub_app(selected_path)
