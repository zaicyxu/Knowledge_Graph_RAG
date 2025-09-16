# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: KnowledgeGraph_RAG
@File Name: main_rag_mlsafety.py
@Software: Python
@Time: Sep/2025
@Author: Rui Xu
@Contact: rxu@kth.se
@Version: 0.1.5
@Description: Streamlit GUI for Neo4j RAG Question Answering System.
              This script provides a user interface to interact with the
              Neo4jRAGSystem defined in main_rag_mlsafety.py.
"""

import streamlit as st
from main_rag_mlsafety import Neo4jRAGSystem
from porlog import configuration


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
    """Clear the stored chat history from session state."""
    st.session_state.history = []


def main() -> None:
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Neo4j RAG QA System", layout="wide")
    st.title("Neo4j RAG Question Answering System")

    # Initialize RAG system
    rag_system = init_rag_system()

    # User input area
    user_input = st.text_area(
        "Enter your question:",
        key="input_area",
        height=100,
        placeholder="Type your question about ML safety, sensors, algorithms..."
    )

    # Action buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Submit"):
            if user_input.strip():
                with st.spinner("Retrieving and generating answer..."):
                    try:
                        answer = rag_system.rag_pipeline(user_input)
                        st.session_state.history = st.session_state.get("history", [])
                        st.session_state.history.append(
                            {"role": "user", "content": user_input}
                        )
                        st.session_state.history.append(
                            {"role": "assistant", "content": answer}
                        )
                        st.session_state.input_area = ""  # Clear input box
                    except Exception as e:
                        st.error(f"Error: {e}")

    with col2:
        if st.button("Clear Chat"):
            clear_chat_history()

    # Display conversation history
    if "history" in st.session_state and st.session_state.history:
        st.markdown("## Conversation History")
        for msg in st.session_state.history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Assistant:** {msg['content']}")


if __name__ == "__main__":
    main()

