# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: Knowledge_safety

@File Name: app.py
@Software: Python
@Time: Sep/2025
@Author: Rui Xu
@Contact: rxu@kth.se
@Version: 0.2.6
@Description: Streamlit GUI for Neo4j RAG Question Answering System.
              This script provides a user interface to interact with the
              Neo4jRAGSystem defined in main_rag_mlsafety.py.

"""

import re
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

    # define canonical labels and their regex to find
    patterns = {
        "final": re.compile(r"(?i)\bfinal\s*answer\b\s*[:：]?", re.MULTILINE),
        "reasoning": re.compile(r"(?i)\breasoning\b\s*[:：]?", re.MULTILINE),
        "safety_implication": re.compile(r"(?i)safety\s+implication[s]?\s*[:：]?", re.MULTILINE),
        "safety_recommendations": re.compile(r"(?i)safety\s+recommendation[s]?\s*[:：]?", re.MULTILINE),
    }

    # find positions of each label in the text
    positions = {}
    for key, pat in patterns.items():
        m = pat.search(text)
        if m:
            positions[key] = m.start()

    # if no labels found, return entire text as final
    if not positions:
        return {"final": text.strip(), "reasoning": "", "safety_implication": "", "safety_recommendations": ""}

    sorted_items = sorted(positions.items(), key=lambda kv: kv[1])
    ordered = [(k, positions[k]) for k, _ in sorted_items]

    # build slices: from each label start to next label start
    parsed = {k: "" for k in patterns.keys()}
    for idx, (key, start) in enumerate(ordered):
        end = len(text)
        if idx + 1 < len(ordered):
            end = ordered[idx + 1][1]
        m = patterns[key].search(text, start)
        if m:
            content = text[m.end():end].strip()
        else:
            content = text[start:end].strip()
        parsed[key] = content

    # return normalized dict (if some keys missing, keep empty)
    return {
        "final": parsed.get("final", ""),
        "reasoning": parsed.get("reasoning", ""),
        "safety_implication": parsed.get("safety_implication", ""),
        "safety_recommendations": parsed.get("safety_recommendations", ""),
    }


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
        for i, entry in enumerate(reversed(st.session_state.history), start=1):
            with st.container():
                st.markdown(f"### Q{i}: {entry['question']}")

                if entry["final"]:
                    st.markdown("**Final Answer:**")
                    lines = [line.strip() for line in entry["final"].splitlines() if line.strip()]
                    if len(lines) > 1:
                        st.markdown("\n".join([f"- {line}" for line in lines]))
                    else:
                        st.code(entry["final"], language="prolog")

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
