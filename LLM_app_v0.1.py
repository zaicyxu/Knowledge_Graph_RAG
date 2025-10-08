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
from collections import defaultdict


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

# Lines that are just code fences (``` or ''') — we'll ignore/remove them
_FENCE_LINE = re.compile(r"(?im)^\s*(?:`{3,}|'{3,}).*$")

def _strip_fence_lines(s: str) -> str:
    """Remove code-fence lines and trim surrounding whitespace."""
    if not s:
        return s
    lines = [ln for ln in s.splitlines() if not _FENCE_LINE.match(ln)]
    return "\n".join(lines).strip()

def parse_response(text: str) -> dict:
    """
    Parse backend text into sections:
      - final: raw block under "Dependency Trace Elements:"
      - prolog_rules: raw block under "Prolog-Based Rules:" / "The Prolog-based Rules:"
      - final_facts: non-empty lines from 'final'
    If labels are absent, put entire text into 'final'.
    """
    out = {"final": "", "final_facts": [], "prolog_rules": ""}

    if not text:
        return out

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1) Dependency Trace Elements block (tolerate fences/blank lines before header and before the next header)
    DEP_BLOCK_RE = re.compile(
        r"""(?imxs)
        ^\s*(?:`{3,}.*\n|\s*'{3,}.*\n|\s*)*          # optional fences/blank lines before header
        Dependency\s*Trace\s*Elements\s*:\s*         # header
        (?P<dep>.*?)                                 # capture block…
        (?=                                          # …until next header (with optional fences) or end
            ^\s*(?:`{3,}.*\n|\s*'{3,}.*\n|\s*)*
            (?:The\s+)?Prolog[-\s]*based\s*Rules\s*:\s*
            | \Z
        )
        """
    )

    # 2) Prolog-Based Rules block (tolerate fences/blank lines before header; capture to end)
    PROLOG_RULES_RE = re.compile(
        r"""(?imxs)
        ^\s*(?:`{3,}.*\n|\s*'{3,}.*\n|\s*)*          # optional fences/blank lines before header
        (?:The\s+)?Prolog[-\s]*based\s*Rules\s*:\s*  # header variants
        (?P<rules>.*)                                # capture to end
        \Z
        """
    )

    m_dep   = DEP_BLOCK_RE.search(text)
    m_rules = PROLOG_RULES_RE.search(text)

    # Fallback: neither header found → keep whole text as 'final'
    if not m_dep and not m_rules:
        final_block = _strip_fence_lines(text.strip())
        out["final"] = final_block
        out["final_facts"] = [ln.strip() for ln in final_block.splitlines() if ln.strip()]
        return out

    dep_block   = _strip_fence_lines(m_dep.group("dep"))     if m_dep   else ""
    rules_block = _strip_fence_lines(m_rules.group("rules")) if m_rules else ""

    # Fill outputs
    out["final"] = dep_block or (text.strip() if not rules_block else "")
    out["prolog_rules"] = rules_block

    # Split dependency block into individual items (non-empty lines)
    if dep_block:
        out["final_facts"] = [ln.strip() for ln in dep_block.splitlines() if ln.strip()]

    return out

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
                        "Related Elements": parsed["final"],
                        "Prolog Rules": parsed["prolog_rules"],
                        "Categoried Elements": parsed["final_facts"],
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
                # st.markdown(f"### Q{i}: {entry['question']}")

                # --- Final Answer as select box (with stable keys) ---
                if entry["Related Elements"]:
                    
                    st.markdown("**Final Answer:**")
                    options = _final_to_options(entry["Related Elements"])

                    if options:
                        selected = st.selectbox(
                            "Choose a related item extracted from Neo4j",
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


                if entry["Prolog Rules"]:
                    st.subheader("Generated Prolog-based Rules")
                    st.code(entry["Prolog Rules"], language="prolog")
                    
                if entry["Categoried Elements"]:
                    categories = defaultdict(list)
                    pattern = re.compile(r"(\w+)\(([^)]+)\)")
                    items = entry["Categoried Elements"]
                    for item in items:
                        match = pattern.search(item)
                        if match:
                            category, name = match.groups()
                            categories[category].append(name.strip())
                    st.subheader("Categorized Elements from Neo4j")
                    selected_category = st.selectbox("Select a category:", list(categories.keys()))

                    # Show corresponding items
                    st.write(f"### Items in category: {selected_category}")
                    st.write(categories[selected_category])
                # Safety Implication

                # Raw backend output
                # with st.expander("Raw backend output"):
                    # st.text(entry["raw"])

                st.markdown("---")


if __name__ == "__main__":
    main()