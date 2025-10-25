# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: Knowledge_Graph_RAG

@File Name: function_call.py
@Software: Python
@Time: Oct/2025
@Author: Rui Xu
@Contact: rxu@kth.se
@Version: 0.1.4
@Description: Implements a robust mechanism to extend a Prolog knowledge
              base using the Function Calling paradigm.
"""


import re
from typing import List, Dict, Any, Optional


# Prolog Knowledge Base Management
class PrologKnowledgeBase:
    """Manages loading and combining Prolog Rules and Facts."""

    def __init__(self, prolog_filepath: str):
        self.prolog_filepath = prolog_filepath
        self.base_rules: str = ""
        self.facts: List[str] = []
        self._load_file()

    def _load_file(self) -> None:
        """Reads the file and separates static rules from existing facts."""
        rule_lines: List[str] = []
        fact_lines: List[str] = []

        with open(self.prolog_filepath, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('%'):
                    continue

                if ':-' in line:
                    rule_lines.append(line)
                else:
                    fact_lines.append(line)

        self.base_rules = "\n".join(rule_lines)
        self.facts = fact_lines

    def add_facts(self, new_fact_list: List[str]) -> None:
        """Appends new template-generated facts to internal fact store."""
        self.facts.extend(new_fact_list)

    def get_full_script(self) -> str:
        """Combines all current facts and the base rules into the final script."""
        return (
            "# === DYNAMIC FACTS (Generated from LLM Keywords) ===\n" +
            "\n".join(self.facts) +
            "\n\n# === STATIC RULES (Loaded once) ===\n" +
            self.base_rules
        )


# Fact Template Generator
class FactGenerator:
    """Creates syntactically correct Prolog fact strings using templates."""

    @staticmethod
    def _format_atom(value: Any) -> str:
        """Handles proper Prolog atom/string quoting."""
        value_str = str(value)
        # Quote if not a simple, all-lowercase atom (letters, digits, underscore)
        if re.match(r"^[a-z0-9_]+$", value_str):
            return value_str
        else:
            return f"'{value_str}'"

    # Component related facts
    def component(self, name: str) -> str:
        return f"component({self._format_atom(name)})."

    def contains(self, component: str, type: str, item: str) -> str:
        return f"contains({self._format_atom(component)}, {type}, {self._format_atom(item)})."

    # Requirement related facts
    def requirement(self, req_id: str) -> str:
        return f"requirement({self._format_atom(req_id)})."

    def requirement_target(self, req_id: str, target: str) -> str:
        return f"requirement_target({self._format_atom(req_id)}, {self._format_atom(target)})."

    def requirement_sensor(self, req_id: str, sensor: str) -> str:
        return f"requirement_sensor({self._format_atom(req_id)}, {self._format_atom(sensor)})."

    def requirement_algorithm(self, req_id: str, algorithm: str) -> str:
        return f"requirement_algorithm({self._format_atom(req_id)}, {self._format_atom(algorithm)})."

    def requirement_model(self, req_id: str, model: str) -> str:
        return f"requirement_model({self._format_atom(req_id)}, {self._format_atom(model)})."


# System Tool Interface
class SystemTools:
    """Functions to process LLM-provided structured data and generate facts."""

    def __init__(self, generator: Optional[FactGenerator] = None):
        self.generator = generator if generator is not None else FactGenerator()

    def add_new_component(self, name: str, sensors: Optional[List[str]] = None,
                          algorithms: Optional[List[str]] = None,
                          actuators: Optional[List[str]] = None) -> List[str]:
        """Creates facts for a new component and its contents."""
        new_facts: List[str] = [self.generator.component(name)]

        for s in (sensors or []):
            new_facts.append(self.generator.contains(name, 'sensor', s))
        for a in (algorithms or []):
            new_facts.append(self.generator.contains(name, 'algorithm', a))
        for act in (actuators or []):
            new_facts.append(self.generator.contains(name, 'actuator', act))

        return new_facts

    def add_new_requirement(self, req_id: str, target: str, sensors: List[str],
                            algorithms: List[str], models: List[str]) -> List[str]:
        """Creates facts for a new requirement and its constraints."""
        new_facts: List[str] = [
            self.generator.requirement(req_id),
            self.generator.requirement_target(req_id, target)
        ]

        for s in sensors:
            new_facts.append(self.generator.requirement_sensor(req_id, s))
        for a in algorithms:
            new_facts.append(self.generator.requirement_algorithm(req_id, a))
        for m in models:
            new_facts.append(self.generator.requirement_model(req_id, m))

        return new_facts


# Application
class PrologScriptGenerator:
    """
    Encapsulates the script generation workflow which previously lived in
    the module-level function `generate_prolog_script`.
    """

    def __init__(self, prolog_file_path: str):
        self.prolog_file_path = prolog_file_path
        self.kb = PrologKnowledgeBase(prolog_file_path)
        self.tools = SystemTools()

    def run(self) -> str:
        """Run the generation process and return the final prolog script (also prints it)."""
        # IStructured keywords generated by the LLM
        llm_keywords_a: Dict[str, Any] = {
            "req_id": "SYS-ML-REQ-5",
            "target": "traffic_light",
            "sensors": ["Mono Camera", "Radar"],
            "algorithms": ["Object Detection"],
            "models": ["YOLOv5"]
        }

        # Execute tool function using keywords
        new_facts_a = self.tools.add_new_requirement(**llm_keywords_a)
        self.kb.add_facts(new_facts_a)

        # Structured keywords generated by the LLM
        llm_keywords_b: Dict[str, Any] = {
            "name": "Radar Processor",
            "sensors": ["Radar"],
            "algorithms": ["Object Tracking", "Tracking Fusion"],
            "actuators": ["throttle"]
        }

        # Execute tool function using keywords
        new_facts_b = self.tools.add_new_component(**llm_keywords_b)
        self.kb.add_facts(new_facts_b)

        # Generate the final combined script
        final_prolog_script = self.kb.get_full_script()

        # Preserve original printing format
        print("=" * 60)
        print("FINAL COMBINED PROLOG SCRIPT")
        print("=" * 60)
        print(final_prolog_script)
        print("=" * 60)

        return final_prolog_script


if __name__ == "__main__":
    PROLOG_FILENAME = "Requirement_Prolog.pl"
    try:
        app = PrologScriptGenerator(PROLOG_FILENAME)
        app.run()
    except FileNotFoundError:
        print(f"Error: File '{PROLOG_FILENAME}' not found. Cannot proceed.")
