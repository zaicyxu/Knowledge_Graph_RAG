# LLM_app_v0.1_plus_select_and_rules.py
# Enhances LLM_app_v0.1 to show:
#   1) "Dependency Trace" as a select box
#   2) "Prolog-based Rules" inside a code box
# You can import the helpers into your existing app, or run this file directly:
#   streamlit run LLM_app_v0.1_plus_select_and_rules.py

import re
import streamlit as st

st.set_page_config(page_title="LLM Dependency Trace & Rules", page_icon="🧠", layout="centered")

# ----------------------------- Helpers ----------------------------------------

def extract_dependency_trace(llm_text: str) -> list[str]:
    """
    Extracts the list under section '4.  Dependency Trace ...' from an LLM response.
    Falls back to section '5.  Final Answer (Dependency-Traced Elements):' if needed.
    Returns a list of cleaned, non-empty lines.
    """
    if not llm_text:
        return []

    # Prefer the explicit "Dependency Trace" block (from 4. ... up to 5.)
    m = re.search(
        r"4\.\s*Dependency\s*Trace.*?:\s*(.+?)\n\s*5\.",
        llm_text, flags=re.DOTALL | re.IGNORECASE
    )
    block = m.group(1) if m else None

    # Fallback: parse the “Final Answer (Dependency-Traced Elements)” block (from 5. ... up to 6.)
    if not block:
        m2 = re.search(
            r"5\.\s*Final\s*Answer\s*\(Dependency-Traced\s*Elements\)\s*:\s*(.+?)\n\s*6\.",
            llm_text, flags=re.DOTALL | re.IGNORECASE
        )
        if m2:
            block = m2.group(1)

    if not block:
        return []

    # Split/clean lines
    lines = [ln.strip() for ln in block.strip().splitlines()]
    cleaned = []
    for ln in lines:
        if ln.startswith("```"):  # ignore code fences
            continue
        ln = re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip()  # remove bullets/numbers
        if ln:
            cleaned.append(ln)
    return cleaned


def extract_prolog_rules(llm_text: str) -> str:
    """
    Extracts the code block under '6.  Prolog-based Rules:'.
    Prefers fenced code between ```prolog ... ```; if not found, returns
    the text after the header up to the end or next numbered section.
    """
    if not llm_text:
        return ""

    # 1) Try to capture fenced prolog code
    m = re.search(
        r"6\.\s*Prolog\-based\s*Rules\s*:\s*```prolog\s*(.+?)\s*```",
        llm_text, flags=re.DOTALL | re.IGNORECASE
    )
    if m:
        return m.group(1).strip()

    # 2) Fallback: anything after the header up to next header or end
    m2 = re.search(
        r"6\.\s*Prolog\-based\s*Rules\s*:\s*(.+)$",
        llm_text, flags=re.DOTALL | re.IGNORECASE
    )
    if m2:
        # Try to stop at a next numbered header
        block = m2.group(1)
        endcut = re.split(r"\n\s*\d+\.\s", block, maxsplit=1)
        return endcut[0].strip()

    return ""

# ------------------------------ UI --------------------------------------------

st.write("## 🧠 LLM Output → Dependency Trace & Prolog Rules")

example_llm = (
    "Okay, I will analyze the safety requirements of the automated driving system (ADS) "
    "described in the knowledge base, focusing on the requirement that \"Object tracks shall be "
    "updated within 100 ms after receiving new sensor data.\"\\n"
    "Analysis:\\n"
    "1.  Key Safety Requirement: ...\\n"
    "2.  Knowledge Base Evidence:\\n"
    "`algorithm(Object Tracking).`: The system utilizes object tracking algorithms.\\n"
    "`sensors(Lidar).`: Lidar sensors are used.\\n"
    "`sensors(Mono Camera).`: Mono cameras are used.\\n"
    "`sensors(Radar).`: Radar sensors are used.\\n"
    "`collect_data(Object Tracking, Mono Camera).`: ...\\n"
    "`collect_data(Object Tracking, TransTrack).`: ...\\n"
    "`serve(Mono Camera, Object Tracking).`: ...\\n"
    "`serve(Object Tracking, Object Tracking).`: ...\\n"
    "`serve(Object Tracking, TransTrack).`: ...\\n"
    "`model(TransTrack).`: ...\\n"
    "`include(Object Tracking, Object Tracking).`: ...\\n"
    "3.  Evaluation: ...\\n"
    "4.  Dependency Trace of listing all directly referenced single elements or entities (Exact Elements by their types):\\n"
    "algorithm(Object Tracking)\\n"
    "sensors(Lidar)\\n"
    "sensors(Mono Camera)\\n"
    "sensors(Radar)\\n"
    "model(TransTrack)\\n"
    "5.  Final Answer (Dependency-Traced Elements):\\n"
    "algorithm(Object Tracking)\\n"
    "sensors(Lidar)\\n"
    "sensors(Mono Camera)\\n"
    "sensors(Radar)\\n"
    "model(TransTrack)\\n"
    "6.  Prolog-based Rules:\\n"
    "```prolog\\n"
    "% Define the requirement\\n"
    "requirement(req_A).\\n"
    "% Define requirement type\\n"
    "requirement_type(req_A, object_tracking_latency).\\n"
    "% Requirement relates to the Object Tracking algorithm\\n"
    "requirement_algorithm(req_A, 'Object Tracking').\\n"
    "% Requirement relates to Lidar sensor\\n"
    "requirement_sensor(req_A, 'Lidar').\\n"
    "% Requirement relates to Mono Camera sensor\\n"
    "requirement_sensor(req_A, 'Mono Camera').\\n"
    "% Requirement relates to Radar sensor\\n"
    "requirement_sensor(req_A, 'Radar').\\n"
    "% Requirement relates to TransTrack model\\n"
    "requirement_model(req_A, 'TransTrack').\\n"
    "% Rules to link requirements with their related elements\\n"
    "req_related_algorithm(Req, Algo) :- requirement(Req), requirement_algorithm(Req, Algo).\\n"
    "req_related_sensor(Req, S) :- requirement(Req), requirement_sensor(Req, S).\\n"
    "req_related_model(Req, M) :- requirement(Req), requirement_model(Req, M).\\n"
    "```"
)

with st.expander("Paste or programmatically set your full LLM response", expanded=True):
    llm_response = st.text_area("LLM Response", value=example_llm, height=300)

# Extract pieces
deps = extract_dependency_trace(llm_response)
prolog = extract_prolog_rules(llm_response)

# Layout: left (select), right (count), then rules box
col1, col2 = st.columns([2, 1])
with col1:
    if deps:
        selected = st.selectbox("Dependency Trace", options=deps, index=0, key="dep_select")
    else:
        selected = None
        st.info("No dependency-traced elements were found.")

with col2:
    st.metric("Count", len(deps))

st.divider()

# Session exposure for downstream apps
if selected:
    st.session_state["selected_dependency_trace_item"] = selected
    st.success(f"Selected: **{selected}**")
    st.code(f"st.session_state['selected_dependency_trace_item'] = '{selected}'", language="python")

# Rules display box
st.subheader("📦 Prolog-based Rules")
if prolog:
    st.code(prolog, language="prolog")
else:
    st.info("No Prolog rules were detected in the response.")

# Debug expanders
with st.expander("🔎 Parsed Dependency Trace (debug)"):
    st.write(deps if deps else "[]")

with st.expander("🔎 Raw Prolog Rules (debug)"):
    st.text(prolog if prolog else "")
