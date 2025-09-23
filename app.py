# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: Knowledge_safety

@File Name: app.py
@Software: Python
@Time: Sep/2025
@Author: Rui Xu
@Contact: rxu@kth.se
@Version: 0.2.2
@Description: Streamlit GUI for Neo4j RAG Question Answering System.
              This script provides a user interface to interact with the
              Neo4jRAGSystem defined in main_rag_mlsafety.py.

"""

# app.py
"""
Streamlit GUI for Neo4j RAG Question Answering System.

This script provides a user interface to interact with the
Neo4jRAGSystem defined in main_rag_mlsafety.py.
"""

import streamlit as st
from main_rag_mlsafety import Neo4jRAGSystem
import configuration


def init_rag_system() -> Neo4jRAGSystem:
    """
    Initialize the RAG system and store it in Streamlit session state
    to avoid repeated Neo4j connections on each interaction.
    """
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = Neo4jRAGSystem(
            uri=configuration.NEO4J_URI,
            user=configuration.NEO4J_USER,
            password=configuration.NEO4J_PASSWORD
        )
    return st.session_state.rag_system


def clear_chat_history() -> None:
    """Clear conversation history stored in session_state."""
    st.session_state.history = []


def main() -> None:
    """Main Streamlit app entry point."""
    st.set_page_config(page_title="Neo4j RAG QA System", layout="wide")
    st.title("Neo4j RAG Question Answering System")

    # initialize rag system and conversation history
    rag_system = init_rag_system()
    if "history" not in st.session_state:
        st.session_state.history = []

    # NOTE:
    # We use a temporary variable for the input text_area (no key assigned).
    # This avoids the Streamlit restriction of modifying session_state[...] for a widget
    # after the widget has been instantiated.
    user_input = st.text_area(
        "Enter your question:",
        value="",
        height=120,
        placeholder="Type your question about ML safety, sensors, algorithms..."
    )

    # Buttons layout
    col1, col2 = st.columns([1, 1])
    with col1:
        # When clicked, run pipeline first, append history, THEN call experimental_rerun()
        # to clear the input area (since it has no key). This guarantees the pipeline runs.
        if st.button("Submit"):
            if user_input.strip():  # avoid empty submissions
                with st.spinner("Retrieving and generating answer..."):
                    try:
                        answer = rag_system.rag_pipeline(user_input)

                        # Append to session history (persistent across reruns)
                        st.session_state.history.append({"role": "user", "content": user_input})
                        st.session_state.history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        # show error to user but do not crash
                        st.error(f"Error while generating answer: {e}")
                # Force a rerun so the text_area resets to its default value (empty)
                st.experimental_rerun()

    with col2:
        if st.button("Clear Chat"):
            clear_chat_history()
            st.experimental_rerun()

    # Display conversation history
    if st.session_state.history:
        st.markdown("## Conversation History")
        for msg in st.session_state.history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Assistant:** {msg['content']}")


if __name__ == "__main__":
    main()
