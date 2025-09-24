# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
Unified Streamlit App:
- Tab 1 (LLM): Ask graph/knowledge questions via Neo4jRAGSystem, parse sections, and surface "Final Answer" as a select box.
- Tab 2 (Graph): Use the selected Final Answer as the Sensor name for Neo4j ➜ (optional Prolog) ➜ Graph exploration originally in test_app_fixed.py.

Assumptions:
- LLM_app_v0.1.py provides parse_response() and a Neo4jRAGSystem in main_rag_mlsafety.
- test_app_fixed.py provides Neo4j helpers and graph rendering utilities.
"""

import os
import re
import hashlib
from typing import Iterable, Tuple, Dict, List, Optional

import streamlit as st

# ---- Import LLM side ----
from main_rag_mlsafety import Neo4jRAGSystem
import configuration

# ---- Import Graph side from your debugged module ----
# We rely on the same functions/classes test_app_fixed.py uses so we don't duplicate logic.
from test_app_fixed import (
    Neo4jConfig,
    get_driver,
    run_cypher_path_query,
    prolog_infer_edges,
    make_nodes_edges,
)
try:
    from streamlit_agraph import agraph, Node, Edge, Config
except Exception:
    agraph = None
    Node = None
    Edge = None
    Config = None


# ======================== LLM helpers ========================

def init_rag_system() -> Neo4jRAGSystem:
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = Neo4jRAGSystem(
            uri=configuration.NEO4J_URI,
            user=configuration.NEO4J_USER,
            password=configuration.NEO4J_PASSWORD,
        )
    return st.session_state.rag_system


def parse_response(text: str) -> dict:
    """(Copied inline for portability)"""
    if not text:
        return {"final": "", "reasoning": "", "safety_implication": "", "safety_recommendations": ""}

    patterns = {
        "final": re.compile(r"(?i)\bfinal\s*answer\b\s*[:：]?", re.MULTILINE),
        "reasoning": re.compile(r"(?i)\breasoning\b\s*[:：]?", re.MULTILINE),
        "safety_implication": re.compile(r"(?i)safety\s+implication[s]?\s*[:：]?", re.MULTILINE),
        "safety_recommendations": re.compile(r"(?i)safety\s+recommendation[s]?\s*[:：]?", re.MULTILINE),
    }
    positions = {}
    for key, pat in patterns.items():
        m = pat.search(text)
        if m:
            positions[key] = m.start()
    if not positions:
        return {"final": text.strip(), "reasoning": "", "safety_implication": "", "safety_recommendations": ""}
    sorted_items = sorted(positions.items(), key=lambda kv: kv[1])
    ordered = [(k, positions[k]) for k, _ in sorted_items]

    parsed = {k: "" for k in patterns.keys()}
    for idx, (key, start) in enumerate(ordered):
        end = len(text)
        if idx + 1 < len(ordered):
            end = ordered[idx + 1][1]
        m = patterns[key].search(text, start)
        content = text[m.end():end].strip() if m else text[start:end].strip()
        parsed[key] = content

    return {
        "final": parsed.get("final", ""),
        "reasoning": parsed.get("reasoning", ""),
        "safety_implication": parsed.get("safety_implication", ""),
        "safety_recommendations": parsed.get("safety_recommendations", ""),
    }


def _final_to_options(final_text: str) -> List[str]:
    if not final_text:
        return []
    lines = [ln.strip() for ln in final_text.splitlines() if ln.strip()]
    if lines:
        return lines
    only = final_text.strip()
    return [only] if only else []


def _stable_key(prefix: str, question: str, raw: str) -> str:
    h = hashlib.sha1((question + "||" + raw).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"


# ======================== UI ========================

def llm_tab() -> Optional[str]:
    """Render LLM Q&A and return the currently selected 'Final Answer' item (to be used as sensor)."""
    rag = init_rag_system()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "selected_sensor" not in st.session_state:
        st.session_state.selected_sensor = None

    st.subheader("LLM: Neo4j RAG Question Answering")
    st.caption("Ask a question. The backend returns structured text (Final Answer / Reasoning / Safety...).")

    with st.form("qa_form"):
        q = st.text_area(
            "Enter your question:",
            value="Which sensor is most related to the pedestrian detection requirement?",
            height=120,
        )
        submitted = st.form_submit_button("Submit")

    if submitted:
        question = q.strip()
        if not question:
            st.warning("Please enter a non-empty question.")
        else:
            with st.spinner("Calling RAG pipeline..."):
                try:
                    raw_answer = rag.rag_pipeline(question)
                    parsed = parse_response(raw_answer)
                    st.session_state.history.append({
                        "question": question,
                        "final": parsed["final"],
                        "reasoning": parsed["reasoning"],
                        "safety_implication": parsed["safety_implication"],
                        "safety_recommendations": parsed["safety_recommendations"],
                        "raw": raw_answer
                    })
                except Exception as e:
                    st.error(f"Error while generating answer: {e}")

    colA, colB = st.columns(2)
    with colA:
        if st.button("Clear LLM History"):
            st.session_state.history = []
            st.session_state.selected_sensor = None

    # Show history and select a Final Answer item
    selected_from_llm: Optional[str] = None
    if st.session_state.history:
        st.markdown("### Conversation History")
        for i, entry in enumerate(reversed(st.session_state.history), start=1):
            with st.container(border=True):
                st.markdown(f"**Q{i}:** {entry['question']}")

                if entry["final"]:
                    st.markdown("**Final Answer (select one item):**")
                    options = _final_to_options(entry["final"])
                    if options:
                        selected = st.selectbox(
                            "Final Answer items",
                            options=options,
                            index=0,
                            key=_stable_key("final_select", entry["question"], entry["raw"]),
                            help="The selected item will be offered to the Graph tab as the Sensor name."
                        )
                        st.caption(f"Selected: {selected}")
                        # propagate last seen selection upward
                        selected_from_llm = selected
                    else:
                        st.info("No discrete items found in final section.")
                        st.code(entry["final"], language="text")

                # Inline non-critical sections
                if entry["reasoning"]:
                    with st.expander("Reasoning"):
                        st.markdown(entry["reasoning"])
                if entry["safety_implication"]:
                    with st.expander("Safety Implication"):
                        st.markdown(entry["safety_implication"])
                if entry["safety_recommendations"]:
                    with st.expander("Safety Recommendations"):
                        st.markdown(entry["safety_recommendations"])

                with st.expander("Raw backend output"):
                    st.text(entry["raw"])
    else:
        st.info("No conversations yet. Ask a question above.")

    # Persist the most recent selection
    if selected_from_llm:
        st.session_state.selected_sensor = selected_from_llm

    return st.session_state.get("selected_sensor")


def graph_tab(sensor_from_llm: Optional[str]) -> None:
    """Render Graph/Prolog tab. Optionally use sensor name from LLM tab."""
    st.subheader("Graph: Neo4j ➜ (optional Prolog) ➜ Visualization")

    # Sidebar for connection & options
    with st.sidebar:
        st.header("Neo4j connection")
        uri = st.text_input("Bolt URI", value=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
        user = st.text_input("User", value=os.getenv("NEO4J_USER", "neo4j"))
        password = st.text_input("Password", type="password", value=os.getenv("NEO4J_PASSWORD", "12345678"))
        database = st.text_input("Database (optional)", value=os.getenv("NEO4J_DB", "")) or None

        st.header("Query")
        use_llm_sensor = st.checkbox("Use Sensor from LLM tab", value=bool(sensor_from_llm))
        default_sensor = sensor_from_llm if (use_llm_sensor and sensor_from_llm) else "Mono Camera"
        sensor = st.text_input("Sensor name", value=default_sensor, disabled=use_llm_sensor and bool(sensor_from_llm))

        include_inferred = st.checkbox("Include Prolog inferred edges", value=False)
        show_tables = st.checkbox("Show tables (edges)", value=False)
        rules_path = st.text_input("Prolog rules file", value="Requirement_Prolog.pl")

        st.divider()
        run_btn = st.button("Run graph query")

    if not run_btn:
        if sensor_from_llm:
            st.info(f"Tip: You have a sensor from the LLM tab: **{sensor_from_llm}**. Click **Run graph query**.")
        else:
            st.info("Tip: Select a Final Answer item in the LLM tab to use it as the Sensor name here.")
        return

    # Run query
    cfg = Neo4jConfig(uri=uri, user=user, password=password, database=database)
    try:
        driver = get_driver(cfg)
    except Exception as e:
        st.error(f"Failed to create Neo4j driver: {e}")
        return

    with driver.session(database=database) as session:
        try:
            base_edges: List[Tuple[str, str, str]] = run_cypher_path_query(session, sensor_name=sensor)
            st.success(f"Fetched {len(base_edges)} base edges from Neo4j.")
        except Exception as e:
            st.error(f"Neo4j step failed: {e}")
            return

    # Build initial graph
    try:
        nodes, edges = make_nodes_edges(base_edges)
    except Exception as e:
        st.error(f"Building graph failed: {e}")
        return

    # Optional Prolog inferences
    inferred_edges: List[Tuple[str, str, str]] = []
    if include_inferred:
        try:
            inferred_edges = prolog_infer_edges(base_edges, rules_path=rules_path)
            st.success(f"Prolog inferred {len(inferred_edges)} edges.")
            # merge inferred
            nodes_inf, edges_inf = make_nodes_edges(inferred_edges)
            edges.extend(edges_inf)
            # nodes update not strictly needed if ids are the same
        except Exception as e:
            st.error(f"Prolog step failed: {e}")

    # Render
    if agraph is None or Config is None:
        st.error("streamlit-agraph is not installed. `pip install streamlit-agraph`")
    else:
        cfg_graph = Config(width=1200, height=600, directed=True, physics=False,
                           nodeHighlightBehavior=True, hierarchical=False)
        st.subheader("Graph (base + inferred)")
        agraph(list(nodes.values()), edges, cfg_graph)

        if show_tables:
            st.subheader("Base edges")
            base_edges = list(set(base_edges))
            st.dataframe([{"source": s, "rel": r, "target": t} for s, r, t in base_edges])
            if include_inferred and inferred_edges:
                inferred_edges = list(set(inferred_edges))
                st.subheader("Inferred edges")
                st.dataframe([{"source": s, "rel": r, "target": t} for s, r, t in inferred_edges])

    st.caption("Tip: Edit Requirement_Prolog.pl to add your own logic. Re-run to see updated inferences.")


def main() -> None:
    st.set_page_config(page_title="Unified LLM + Graph App", layout="wide")
    st.title("Unified LLM ➜ Sensor ➜ Graph App")

    tabs = st.tabs(["LLM (Final Answer ➜ Sensor)", "Graph (Neo4j + Prolog)"])
    with tabs[0]:
        sensor_from_llm = llm_tab()
    with tabs[1]:
        graph_tab(sensor_from_llm)


if __name__ == "__main__":
    main()
