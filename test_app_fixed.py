#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit + Neo4j + Prolog (pyswip) demo app

- Queries Neo4j for paths from a selected Sensor to ML_Safety_Requirement
- Optionally runs Prolog rules to infer extra edges
- Visualizes base + inferred graph with streamlit-agraph
- Shows tabular views

This file is a *debugged & self-contained* version of your previous test_app.py,
removing duplicate headers and ellipsis placeholders, adding missing imports,
and hardening error handling so the app fails gracefully when optional
dependencies are absent.

Requirements on your machine (install in your env):
    pip install streamlit neo4j streamlit-agraph pyswip

Run locally:
    streamlit run test_app.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional,Any

# --- Third-party imports (all optional except streamlit) ----------------------
try:
    import streamlit as st
except Exception as e:
    print("ERROR: streamlit is required to run this app.", file=sys.stderr)
    raise

# Neo4j Python driver
try:
    from neo4j import GraphDatabase, Driver, Session
except Exception:
    GraphDatabase = None  # type: ignore
    Driver = None  # type: ignore
    Session = None  # type: ignore

# Graph visualization
try:
    from streamlit_agraph import agraph, Node, Edge, Config
except Exception:
    agraph = None
    Node = None
    Edge = None
    Config = None

# Prolog (pyswip)
try:
    from pyswip import Prolog  # type: ignore
except Exception:
    Prolog = None  # type: ignore


# === Configuration dataclasses ===============================================

@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: Optional[str] = None


# === Neo4j helpers ============================================================

def get_driver(cfg: Neo4jConfig) -> Driver:
    if GraphDatabase is None:
        raise RuntimeError("neo4j driver not installed. `pip install neo4j`.")
    auth = (cfg.user, cfg.password)
    return GraphDatabase.driver(cfg.uri, auth=auth)


def run_cypher_path_query(session: Session, sensor_name: str) -> List[Tuple[str, str, str]]:
    """Return edges as (source, rel_type, target) from a variable-length path query."""
    cypher = """
    MATCH p = (m:Sensors {name: $sensor})-[*]->(n:ML_Safety_Requirement)
    WITH nodes(p) AS nds, relationships(p) AS rels
    UNWIND range(0, size(rels)-1) AS i
    RETURN
      nds[i].name             AS source,
      type(rels[i])           AS rel_type,
      nds[i+1].name           AS target
    ORDER BY i
    """
    rows = session.run(cypher, sensor=sensor_name)
    return [(r["source"], r["rel_type"], r["target"]) for r in rows]


# === Prolog helpers ===========================================================

def prolog_load_rules(file_path: str) -> Optional[object]:
    """Load Prolog knowledge base if pyswip is available, else None."""
    if Prolog is None:
        return None
    pl = Prolog()
    pl.consult(file_path)
    return pl


def prolog_find_inferred(pl: object) -> List[Tuple[str, str, str]]:
    """
    Example: ask Prolog for inferred edges as triples inferred(S,Rel,T).
    Customize to match your Requirement_Prolog.pl rules.
    """
    if pl is None:
        return []
    inferred = []
    try:
        q = pl.query("inferred(S,R,T)")
        for sol in q:  # type: ignore[attr-defined]
            inferred.append((str(sol["S"]), str(sol["R"]), str(sol["T"])))
    except Exception:
        # If the predicate doesn't exist, just return empty
        return []
    return inferred


def prolog_infer_edges(prolog,initial_values) -> List[Tuple[str, str, str]]:
    """
    Query predicates that exist in Requirement_Modify.* and convert to labeled edges.
    Returns a list of (source, relation, target).
    """
    inferred = []
    initial_list = [initial_values]
    def _q(query: str) -> List[Dict[str, Any]]:
        try:
            return list(prolog.query(query))
        except Exception:
            return []
    
    def _relationship_investigate( search_name:list, query_template:str,keyword:str,edge_name:str )-> List[str]:
        temporal_list = []
        try:
            for values in search_name:
                query = query_template.format(values)
                # print (query_name)
                for sol in _q(query):
                    # print (query_name)
                    inferred.append((str(sol[keyword]), edge_name, values))
                    temporal_list.append(str(sol[keyword]))
        except Exception:
                print ("The search list is empty!")
        # initial_list = 
        temporal_list = list(set(temporal_list))
        return temporal_list
        
    requirement_check = "req_related_sensor(Req, '{}')"
    requirement_list = _relationship_investigate(initial_list,requirement_check,"Req","REQ_SENSOR")
    
    component_check = "traces_to_requirement(C, '{}')"
    component_list = _relationship_investigate(requirement_list,component_check,"C","TRACES_TO")    
    
    
    algorithm_check = "req_related_algorithm('{}', A)"
    algorithm_list = _relationship_investigate(requirement_list,algorithm_check,"A","REQ_ALG")  
    
    model_check = "req_related_model('{}', M)"
    model_list = _relationship_investigate(requirement_list,model_check,"M","REQ_MODEL")  
    
    # requirement_list = []
    # Requirement -> Sensor / Algorithm / Model
    # for sol in _q(f"req_related_sensor(Req, '{values}')"):
    #     inferred.append((str(sol["Req"]), "REQ_SENSOR", values))
    #     requirement_list.append(str(sol["Req"]))
        
    # Component -> Requirement traces
    # for sol in _q("traces_to_requirement(C, Req)"):
    #     inferred.append((str(sol["C"]), "TRACES_TO", str(sol["Req"])))
        
    # for sol in _q("req_related_algorithm(Req, A)"):
    #     inferred.append((str(sol["Req"]), "REQ_ALG", str(sol["A"])))
    # for sol in _q("req_related_model(Req, M)"):
    #     inferred.append((str(sol["Req"]), "REQ_MODEL", str(sol["M"])))

    # # Component satisfies exact requirement elements
    # for sol in _q("comp_req_sensor(C, Req, S)"):
    #     inferred.append((str(sol["C"]), "SAT_SENSOR", str(sol["S"])))
    # for sol in _q("comp_req_algorithm(C, Req, A)"):
    #     inferred.append((str(sol["C"]), "SAT_ALG", str(sol["A"])))
    # for sol in _q("comp_req_model(C, Req, M)"):
    #     inferred.append((str(sol["C"]), "SAT_MODEL", str(sol["M"])))
        
    # if len(requirement_list!=0):
    #     # Component -> Requirement traces
    #     for requirement in requirement_list:
    #         for sol in _q(f"traces_to_requirement(C, '{requirement}')"):
    #             inferred.append((str(sol["C"]), "TRACES_TO", str(sol["Req"])))
                
            # for sol in _q("req_related_algorithm(Req, A)"):
            #     inferred.append((str(sol["Req"]), "REQ_ALG", str(sol["A"])))
            # for sol in _q("req_related_model(Req, M)"):
            #     inferred.append((str(sol["Req"]), "REQ_MODEL", str(sol["M"])))
    
            # # Component satisfies exact requirement elements
            # for sol in _q("comp_req_sensor(C, Req, S)"):
            #     inferred.append((str(sol["C"]), "SAT_SENSOR", str(sol["S"])))
            # for sol in _q("comp_req_algorithm(C, Req, A)"):
            #     inferred.append((str(sol["C"]), "SAT_ALG", str(sol["A"])))
            # for sol in _q("comp_req_model(C, Req, M)"):
            #     inferred.append((str(sol["C"]), "SAT_MODEL", str(sol["M"])))

    return inferred

def prolog_query_pairs(pl: object, goal: str, vars_: Tuple[str, str]) -> List[Tuple[str, str]]:
    """Utility: run a two-var query like 'fof(A,C)' -> returns list of (A,C)."""
    if pl is None:
        return []
    results: List[Tuple[str, str]] = []
    try:
        q = pl.query(goal)  # type: ignore[attr-defined]
        for sol in q:
            results.append((str(sol[vars_[0]]), str(sol[vars_[1]])))
    except Exception:
        pass
    return results


# === Streamlit UI =============================================================

def make_nodes_edges(triples: Iterable[Tuple[str, str, str]]) -> Tuple[Dict[str, Node], List[Edge]]:
    """Create unique Node objects and Edge objects for agraph."""
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []
    for s, rel, t in triples:
        if s not in nodes:
            nodes[s] = Node(id=s, label=s, size=15)
        if t not in nodes:
            nodes[t] = Node(id=t, label=t, size=15)
        edges.append(Edge(source=s, target=t, label=rel))
    return nodes, edges


def main() -> None:
    st.set_page_config(page_title="Neo4j + Prolog Graph", layout="wide")

    st.title("Neo4j ➜ Prolog reasoning ➜ Graph")
    st.caption("Query Neo4j, optionally enrich with Prolog inferences, and visualize.")

    with st.sidebar:
        st.header("Neo4j connection")
        uri = st.text_input("Bolt URI", value=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
        user = st.text_input("User", value=os.getenv("NEO4J_USER", "neo4j"))
        password = st.text_input("Password", type="password", value=os.getenv("NEO4J_PASSWORD", "12345678"))
        database = st.text_input("Database (optional)", value=os.getenv("NEO4J_DB", "")) or None

        st.header("Query")
        sensor = st.text_input("Sensor name", value="Mono Camera")
        include_inferred = st.checkbox("Include Prolog inferred edges", value=False)
        rules_path = st.text_input("Prolog file", value="Requirement_Prolog.pl")
        show_tables = st.checkbox("Show tables", value=True)

        run_btn = st.button("Run")

    if run_btn:
        # --- Neo4j base edges --------------------------------------------------
        base_edges: List[Tuple[str, str, str]] = []
        try:
            cfg = Neo4jConfig(uri=uri, user=user, password=password, database=database)
            driver = get_driver(cfg)
            with driver.session(database=database) as session:
                base_edges = run_cypher_path_query(session, sensor_name=sensor)
            st.success(f"Fetched {len(base_edges)} base edges from Neo4j.")
        except Exception as e:
            st.error(f"Neo4j step failed: {e}")
            st.stop()

        # Build initial graph
        nodes, edges = make_nodes_edges(base_edges)

        # --- Prolog reasoning (optional) --------------------------------------
        inferred_edges: List[Tuple[str, str, str]] = []
        if include_inferred:
            try:
                if not os.path.exists(rules_path):
                    st.warning(f"Prolog file not found: {rules_path}. Skipping inference.")
                else:
                    pl = prolog_load_rules(rules_path)
                    if pl is None:
                        st.info("pyswip not installed or SWI-Prolog not available. Skipping inference.")
                    else:
                        inferred_edges = prolog_infer_edges(pl,sensor)
                        # merge into graph
                        for s, r, t in inferred_edges:
                            if s not in nodes:
                                nodes[s] = Node(id=s, label=s, size=15)
                            if t not in nodes:
                                nodes[t] = Node(id=t, label=t, size=15)
                            edges.append(Edge(source=s, target=t, label=r,dashes=True,color="red", width=2,))
                        st.success(f"Added {len(inferred_edges)} inferred edges from Prolog.")
            except Exception as e:
                st.error(f"Prolog reasoning failed: {e}")

        # --- Visualization -----------------------------------------------------
        if agraph is None or Config is None:
            st.error("streamlit-agraph is not installed. `pip install streamlit-agraph`")
        else:
            cfg = Config(width=1200, height=600, directed=True, physics=False,
                         nodeHighlightBehavior=True, hierarchical=False)
            st.subheader("Graph (base + inferred)")
            agraph(list(nodes.values()), edges, cfg)

            if show_tables:
                st.subheader("Base edges")
                base_edges = list(set(base_edges))
                st.dataframe([{"source": s, "rel": r, "target": t} for s, r, t in base_edges])
                if include_inferred and inferred_edges:
                    inferred_edges = list(set(inferred_edges))
                    st.subheader("Inferred edges")
                    st.dataframe([{"source": s, "rel": r, "target": t} for s, r, t in inferred_edges])

    st.caption("Tip: Edit Requirement_Prolog.pl to add your own logic. Re-run to see updated inferences.")


if __name__ == "__main__":
    main()
