# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: Knowledge_safety

@File Name: app.py
@Software: Python
@Time: Sep/2025
@Author: Rui Xu
@Contact: rxu@kth.se
@Version: 0.2.7
@Description: Streamlit GUI for Neo4j RAG Question Answering System.
              This script provides a user interface to interact with the
              Neo4jRAGSystem defined in main_rag_mlsafety.py.

              v0.2.7: "Final Answer" lines are rendered as selectable
              checkboxes so users can pick results for downstream tasks.
              Selections are accumulated in st.session_state['selected_results'].
"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python


import re
import hashlib  # <-- NEW: for stable widget keys
import streamlit as st
from main_rag_mlsafety import Neo4jRAGSystem
import configuration


def init_rag_system() -> Neo4jRAGSystem:
    """Initialize and cache RAG system in session state to avoid reconnecting each run."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = Neo4jRAGSystem(
            uri=configuration.NEO4J_URI,
            user=configuration.NEO4J_USER,
            password=configuration.NEO4J_PASSWORD,
        )
    return st.session_state.rag_system


def clear_chat_history() -> None:
    """Clear conversation history from session_state."""
    st.session_state.history = []


def parse_response(text: str) -> dict:
    """
    Parse backend text into sections:
      - final
      - reasoning
      - safety_implication
      - safety_recommendations
    If labels are absent, put entire text into 'final'.
    This function is robust to case and small label variations.
    """
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


def _final_to_options(final_text: str) -> list[str]:
    """
    Turn the 'final' section into a list of options for a select box.
    - Splits on lines and trims empties.
    - If that yields nothing but there is content, use the whole text as one option.
    """
    if not final_text:
        return []
    # basic line split
    lines = [ln.strip() for ln in final_text.splitlines() if ln.strip()]
    if lines:
        return lines
    # fallback: one big option
    only = final_text.strip()
    return [only] if only else []


def _stable_key(prefix: str, entry: dict) -> str:
    """
    Produce a stable widget key for a history entry, independent of visual order.
    Uses a short hash of question + raw backend output.
    """
    h = hashlib.sha1((entry.get("question", "") + "||" + entry.get("raw", "")).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"


def main() -> None:
    """Main Streamlit app entry point."""
    st.set_page_config(page_title="Neo4j RAG QA System", layout="wide")
    st.title("Neo4j RAG Question Answering System (Structured Display)")

    # initialize
    rag_system = init_rag_system()
    if "history" not in st.session_state:
        st.session_state.history = []

    st.markdown(
        "Ask graph questions. Backend returns a combined text (Final Answer + Reasoning + Safety...). "
        "This UI will parse and display these parts separately."
    )

    with st.form("qa_form"):
        user_input = st.text_area(
            "Enter your question:",
            value="",
            height=120,
            placeholder="e.g. Which kind of algorithm has been used for ML_Safety_Requirement by the sensors of mono camera?"
        )
        submitted = st.form_submit_button("Submit")

    if submitted:
        question = user_input.strip()
        if not question:
            st.warning("Please enter a non-empty question.")
        else:
            with st.spinner("Calling RAG pipeline..."):
                try:
                    raw_answer = rag_system.rag_pipeline(question)
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

    if st.button("Clear Chat History"):
        clear_chat_history()

    if st.session_state.history:
        st.markdown("## Conversation History")
        # Render newest first, but keys are now stable so reruns won't break widgets
        for i, entry in enumerate(reversed(st.session_state.history), start=1):
            with st.container():
                st.markdown(f"### Q{i}: {entry['question']}")

                # --- Final Answer as select box (with stable keys) ---
                if entry["final"]:
                    st.markdown("**Final Answer:**")
                    options = _final_to_options(entry["final"])

                    if options:
                        selected = st.selectbox(
                            "Choose a final answer item",
                            options=options,
                            index=0,
                            key=_stable_key("final_select", entry)
                        )
                        # Always show the selected content clearly below the dropdown
                        st.success(selected)
                    else:
                        # Nothing parsed into options; show raw final safely
                        st.info("No discrete items found in the final section.")
                        st.code(entry["final"], language="text")

                # Reasoning
                if entry["reasoning"]:
                    with st.expander("Reasoning"):
                        st.markdown(entry["reasoning"])

                # Safety Implication
                if entry["safety_implication"]:
                    with st.expander("Safety Implication"):
                        st.markdown(entry["safety_implication"])

                # Safety Recommendations
                if entry["safety_recommendations"]:
                    with st.expander("Safety Recommendations"):
                        st.markdown(entry["safety_recommendations"])

                # Raw backend output
                with st.expander("Raw backend output"):
                    st.text(entry["raw"])

                st.markdown("---")


if __name__ == "__main__":
    main()