# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: Knowledge_safety

@File Name: main_rag_mlsafety.py
@Software: Python
@Time: Sep/2025
@Author: Rui Xu
@Contact: rxu@kth.se
@Version: 0.2.4
@Description: Based on the original RAG system, the output format was modified to conform to the
              Prolog fact and rules format standards for subsequent processing.
"""

import json
import pickle
import re
import numpy as np
from neo4j import GraphDatabase
import google.generativeai as genai
import configuration
from sklearn.decomposition import PCA

# Configure Gemini API
genai.configure(api_key=configuration.GEMINI_API_KEY)
model = genai.GenerativeModel(configuration.GENERATION_MODEL)


class Neo4jRAGSystem:
    def __init__(self, uri, user, password, cache_file=configuration.CACHE_FILE):
        self.driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=60)
        self.cache_file = cache_file
        self.embedding_cache = self.load_cache()

    def close(self):
        self.driver.close()

    def load_cache(self):
        """Load embedding cache from pickle file."""
        try:
            with open(self.cache_file, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return {}

    def save_cache(self):
        """Save embedding cache to pickle file."""
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.embedding_cache, f)

    def cosine_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two vectors.
        Ensure both vectors are NumPy arrays before computation.
        """
        if isinstance(vec1, str):
            try:
                vec1 = json.loads(vec1)
            except json.JSONDecodeError:
                print(f"[ERROR] Failed to decode embedding: {vec1}")
                return 0.0
        if isinstance(vec2, str):
            try:
                vec2 = json.loads(vec2)
            except json.JSONDecodeError:
                print(f"[ERROR] Failed to decode embedding: {vec2}")
                return 0.0

        vec1 = np.array(vec1, dtype=np.float32)
        vec2 = np.array(vec2, dtype=np.float32)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def update_embeddings(self):
        """Update missing embeddings for nodes in the Neo4j database."""
        with self.driver.session() as session:
            # inside update_embeddings()
            cypher_query = """
                MATCH (n)
                WHERE any(lbl IN labels(n) WHERE lbl IN $target_labels)
                  AND (n.embedding IS NULL OR n.embedding = [])
                RETURN elementId(n) AS id, n.name AS name, elementId(n) AS node_id
            """
            target_labels = configuration.TARGET_LABELS

            records = session.run(cypher_query, target_labels=target_labels)

            for record in records:
                text = record['name']  # Use 'name' field instead of 'title'+'body_markdown'
                embedding = self.get_embedding(text)
                if embedding:
                    update_query = """
                    MATCH (n) WHERE elementId(n) = $node_id
                    SET n.embedding = $embedding
                    """
                    session.run(update_query, node_id=record['node_id'], embedding=embedding)

            print("Embeddings updated successfully.")

    def get_embedding(self, text):
        """Generate embedding using Gemini API or fetch from cache."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        try:
            response = genai.embed_content(
                model=configuration.EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_query"
            )
            embedding = response["embedding"]
            self.embedding_cache[text] = embedding
            self.save_cache()
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def extract_keywords(self, query_text, similarity_threshold=configuration.KEYWORD_SIMILARITY_THRESHOLD):
        tokens = re.findall(r'\b\w+\b', query_text)
        tokens = [t for t in tokens if
                  t.lower() not in {"the", "a", "an", "is", "are", "what", "which", "how", "does", "do", "to", "in",
                                    "of", "and"}]
        tokens = [t for t in tokens if len(t) >= 1]  # allow single-char tokens (useful for ML etc.)
        # add bigrams
        bigrams = [' '.join([tokens[i], tokens[i + 1]]) for i in range(len(tokens) - 1)]
        candidates = list(dict.fromkeys(tokens + bigrams))  # unique preserve order

        q_emb = self.get_embedding(query_text)
        if q_emb is None:
            return [c.lower() for c in candidates[:5]]

        scored = []
        for c in candidates:
            emb = self.get_embedding(c)
            if emb is None:
                continue
            sim = self.cosine_similarity(emb, q_emb)
            scored.append((c, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        if not scored:
            return [c.lower() for c in candidates[:5]]
        # return top items with sim >= threshold OR top 5 as fallback
        kws = [w for w, s in scored if s >= similarity_threshold]
        return [w.lower() for w in (kws if kws else [x[0] for x in scored[:5]])]

    def compute_semantic_diversity(self, query_text):
        """
        Compute the semantic diversity of a given query by analyzing the variance in word embeddings.
        A higher variance indicates a more diverse semantic meaning in the query.
        """
        stopwords = configuration.STOP_WORDS
        words = [word for word in re.findall(r'\b\w+\b', query_text.lower())
                 if word not in stopwords]

        if len(words) < configuration.MIN_WORDS_FOR_SEMANTIC_ANALYSIS:
            return 0.4

        # Retrieve embeddings for up to 10 words.
        embeddings = np.array([self.get_embedding(word) for word in words[:configuration.MAX_WORDS_FOR_EMBEDDING]])

        # Apply PCA to reduce dimensionality and analyze variance.
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        variance = np.sum(np.var(reduced, axis=0))

        # Normalize variance using a maximum expected variance value.
        max_variance = configuration.MAX_VARIANCE
        return min(variance / max_variance, 1.0)

    def determine_query_depth(self, query_text):
        """
        Determine the appropriate query depth based on semantic and syntactic complexity.
        """
        semantic_score = self.compute_semantic_diversity(query_text)
        syntax_score = self.compute_syntactic_complexity(query_text)

        # Weighted combination of semantic and syntactic complexity.
        combined_score = (configuration.SEMANTIC_WEIGHT * semantic_score +
                          configuration.SYNTACTIC_WEIGHT * syntax_score)

        if combined_score < 0.3:
            depth = 1
        elif 0.3 <= combined_score < 0.6:
            depth = 2
        elif 0.6 <= combined_score < 0.8:
            depth = 3
        else:
            depth = 4

        if re.search(r'\b(indirect|through|chain|subsidiary)\b', query_text, re.I):
            depth = max(depth, 3)

        return depth

    def compute_syntactic_complexity(self, query_text):
        """
        Compute syntactic complexity based on sentence structure.
        """
        words = query_text.split()

        # Normalize sentence length to a max of 15 words.
        length_score = min(len(words) / 15, 1.0)

        wh_words = {configuration.WH_WORDS}
        wh_score = min(sum(1 for word in words if word.lower() in wh_words) / 2, 1.0)

        # Count clauses
        clause_count = len(re.findall(r',|;| but | however | although ', query_text))
        clause_score = min(clause_count / 2, 1.0)

        # Compute weighted complexity score.
        return (length_score * configuration.LENGTH_WEIGHT +
                wh_score * configuration.WH_WORD_WEIGHT +
                clause_score * configuration.CLAUSE_WEIGHT)


##### This function is not used #####

    # def analyze_query_intent_with_LLM(self, query_text):
    #     """
    #     Parse query intent through LLM to extract target entities, relationships, and filter conditions.
    #     Optimized for the specific MLSafety knowledge graph structure.
    #     """

    #     system_prompt = """
    #                     You are an AI assistant specialized in analzying the system models structured as Neo4j.
                    
    #                     DOMAIN CONTEXT:
    #                     - This is a knowledge base focusing on safety-critical systems
    #                     - Core Elements: Sensors, Algorithms, Functionalities, Models, components
    #                     - Data flow: Sensors → Collect_Data → algorithms → ML_Flow → Safety_Requirements
                    
    #                     ENTITY HIERARCHY:
    #                     1. System Level: System_Description, System_Safety_Requirement
    #                     2. ML Pipeline: ML_Flow, algorithms, Sensors, actuators
    #                     3. Safety Requirements: ML_Safety_Requirement, functional, functionalility
    #                     4. Components: Sensors, actuators, algorithms
                    
    #                     RELATIONSHIP SEMANTICS:
    #                     - NEXT: Sequential flow between ML_Flow components
    #                     - Input/Output: Data flow direction
    #                     - Consist/Include: Composition relationships
    #                     - Serve: Functional serving relationships
    #                     - Collect_Data: Sensor data collection
                    
    #                     QUERY PATTERN EXAMPLES:
    #                     - "Which sensors feed data to anomaly detection flow?" → Sensors + Collect_Data + ML_Flow
    #                     - "What safety requirements apply to the prediction algorithm?" → ML_Safety_Requirement + algorithms
    #                     - "Show the ML flow sequence for system X" → ML_Flow + NEXT relationships
                    
    #                     Extract query intent in JSON format:
    #                     {
    #                         "entities": ["primary_entity", "secondary_entity"],
    #                         "relationships": ["key_relationship", "supporting_relationship"],
    #                         "filters": {"property": "value", "name_contains": "keyword"}
    #                     }
                    
    #                     Focus on safety and data flow aspects of the query.
    #                     """

    #     response = model.generate_content(system_prompt + "\nUser question: " + query_text)

    #     try:
    #         return json.loads(response.text)
    #     except Exception:
    #         return {"entities": [], "relationships": [], "filters": {}}
#################################################################################

    def retrieve_relevant_entities(self, query_text, top_k=configuration.TOP_K_RESULTS,
                                   similarity_threshold=configuration.SIMILARITY_THRESHOLD,
                                   depth=configuration.DEFAULT_QUERY_DEPTH):
        """
        Retrieve relevant entities from Neo4j and return them in the expected schema.
        """
        keywords = self.extract_keywords(query_text)
        if not keywords:
            print("[WARN] No keywords extracted, using similarity-based retrieval only.")
            keywords = []

        # get labels in DB
        with self.driver.session() as session:
            try:
                result = session.run("CALL db.labels()")
                existing_labels = {record["label"] for record in result}
            except Exception as e:
                print(f"[ERROR] Could not fetch db labels: {e}")
                existing_labels = set()

        desired_labels = configuration.DESIRED_LABELS
        selected_labels = desired_labels.intersection(existing_labels)
        if not selected_labels:
            print("[WARN] No desired labels matched db.labels(). Using all existing labels for search.")
            selected_labels = existing_labels if existing_labels else desired_labels

        # Keyword-based Cypher
        cypher_template = """
        MATCH p = (n)-[*1..{DEPTH}]-(m)
        WHERE any(kw IN $keywords WHERE any(x IN nodes(p) WHERE (
                (x.name IS NOT NULL AND toLower(x.name) CONTAINS kw)
                OR
                (x.type IS NOT NULL AND toLower(x.type) CONTAINS kw)
                OR
                any(lbl IN labels(x) WHERE toLower(lbl) CONTAINS kw)
            )))
        WITH p, nodes(p) AS np
        UNWIND [x IN np WHERE any(kw IN $keywords WHERE (
                (x.name IS NOT NULL AND toLower(x.name) CONTAINS kw)
                OR
                (x.type IS NOT NULL AND toLower(x.type) CONTAINS kw)
                OR
                any(lbl IN labels(x) WHERE toLower(lbl) CONTAINS kw)
            ))] AS anchor
        RETURN DISTINCT
            elementId(anchor) AS node1_id,
            anchor.name AS node1_name,
            labels(anchor) AS node1_type,
            [rel IN relationships(p) | type(rel)] AS relations,
            [x IN np | {id: elementId(x), name: x.name, type: labels(x)}] AS connected_nodes
        LIMIT 200
        """
        cypher_query = cypher_template.replace("{DEPTH}", str(depth)).replace(
            "{MAX_RESULTS}", str(configuration.MAX_RETRIEVAL_RESULTS))

        keyword_entities = []
        if keywords:
            with self.driver.session() as session:
                try:
                    raw = session.run(cypher_query, {"keywords": [kw.lower() for kw in keywords]}).data()
                    for rec in raw:
                        node1 = {
                            "id": rec.get("node1_id"),
                            "name": rec.get("node1_name"),
                            "type": rec.get("node1_type")
                        }
                        conn = rec.get("connected_nodes") or []
                        relations = rec.get("relations") or []
                        keyword_entities.append({
                            "node1": node1,
                            "relations": relations,
                            "connected_nodes": conn
                        })
                except Exception as e:
                    print(f"[ERROR] Keyword search failed: {e}")
                    keyword_entities = []

        if keyword_entities:
            print(f"[INFO] Keyword-based match found: {len(keyword_entities)} entities.")
            return keyword_entities

        # --- Similarity fallback ---
        print("[INFO] No keyword-based match found. Falling back to similarity-based retrieval...")

        query_embedding = self.get_embedding(query_text)
        if query_embedding is None:
            print("[ERROR] Failed to generate query embedding.")
            return []

        similarity_entities = []
        seen_ids = set()

        with self.driver.session() as session:
            for label in selected_labels:
                try:
                    res = session.run(
                        f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL RETURN elementId(n) AS id, n.name AS name, n.embedding AS embedding, labels(n) AS labels LIMIT 100"
                    )
                    for record in res:
                        rid = record["id"]
                        name = record.get("name")
                        emb = record.get("embedding")
                        if emb is None:
                            continue
                        if isinstance(emb, str):
                            try:
                                emb = json.loads(emb)
                            except Exception:
                                continue
                        sim = self.cosine_similarity(query_embedding, emb)
                        print(f"[DEBUG] sim({name}) = {sim:.4f}")
                        if sim >= similarity_threshold and rid not in seen_ids:
                            # fetch neighbours/relations for this node to build connected_nodes
                            try:
                                neighbor_q = session.run(
                                    """
                                    MATCH (n)
                                    WHERE elementId(n) = $id
                                    OPTIONAL MATCH (n)-[r]-(m)
                                    RETURN collect(DISTINCT {id: elementId(m), name: m.name, type: labels(m)}) AS connected_nodes,
                                           collect(DISTINCT type(r)) AS relations,
                                           labels(n) AS node_labels,
                                           n.name AS node_name
                                    """,
                                    {"id": rid}
                                ).single()
                                conn_nodes = neighbor_q["connected_nodes"] or []
                                relations = neighbor_q["relations"] or []
                                node_labels = neighbor_q["node_labels"] or record.get("labels")
                                node_name = neighbor_q["node_name"] or name
                            except Exception as e:
                                print(f"[WARN] Could not fetch neighbours for node id {rid}: {e}")
                                conn_nodes = []
                                relations = []
                                node_labels = record.get("labels")
                                node_name = name

                            entity = {
                                "node1": {"id": rid, "name": node_name, "type": node_labels},
                                "relations": relations,
                                "connected_nodes": conn_nodes
                            }
                            similarity_entities.append(entity)
                            seen_ids.add(rid)
                except Exception as e:
                    print(f"[ERROR] Similarity search failed for label {label}: {e}")
                    continue

        if similarity_entities:
            print(f"[INFO] Similarity fallback found {len(similarity_entities)} results.")
            return similarity_entities

        print("[INFO] Similarity fallback found 0 additional results.")
        return []

    def _to_prolog_atom(self, text, fallback=None):
        """
        Convert a given value into a safe Prolog atom.
        """
        # Choose fallback if text is None
        if text is None:
            if fallback is None:
                return "unknown"
            text = str(fallback)

        # Ensure string
        if not isinstance(text, str):
            text = str(text)

        # basic normalization
        raw = text.strip()
        if raw == "":
            raw = fallback if fallback is not None else "unknown"
            raw = str(raw)

        # safe token: lowercase, spaces -> underscores, non-alnum -> underscore
        token = raw.lower().replace(" ", "_")
        token = re.sub(r'[^a-z0-9_]', '_', token)

        # ensure it starts with letter or underscore for bare atom
        if re.match(r'^[a-z_]\w*$', token):
            return token

        # fallback: return quoted atom, escape single quotes inside
        escaped = raw.replace("'", "\\'")
        return f"'{escaped}'"

    def _label_to_prolog_predicate(self, label):
        """
        Convert a graph label or relation name into a safe Prolog predicate name.
        - Lowercases and replaces non-alphanumeric chars with underscore.
        - If result doesn't start with a letter/underscore, returns a quoted fallback.
        Minimal and reversible transformation; keeps names human-readable.
        """
        import re
        if not label:
            return "unknown"
        s = str(label).strip().lower().replace(" ", "_")
        s = re.sub(r'[^a-z0-9_]', '_', s)
        # if it doesn't form a valid bare atom name, keep it quoted as fallback
        if not re.match(r'^[a-z_]\w*$', s):
            # quoted atom - preserve original label inside quotes
            return f"'{label}'"
        return s

    def generate_answer(self, user_question, relevant_entities):
        """
        Generate an answer using Gemini LLM with optimized prompt engineering.
        """
        if not relevant_entities:
            return "% No facts found.\n"

        keywords = self.extract_keywords(user_question)

        # Generate Prolog facts from retrieved entities
        prolog_facts = []
        for entity in relevant_entities:
            try:
                node1 = entity.get('node1', {})
                node1_id = node1.get('id')
                raw_n1_name = node1.get('name') if node1.get('name') is not None else f"node_{node1_id}"
                e1_name = self._to_prolog_atom(raw_n1_name, fallback=f"node_{node1_id}")

                # Extract primary label
                raw_e1_label = None
                if node1.get('type'):
                    t = node1.get('type')
                    if isinstance(t, (list, tuple)) and len(t) > 0:
                        raw_e1_label = t[0]
                    else:
                        raw_e1_label = t
                e1_type = self._label_to_prolog_predicate(raw_e1_label)
                # prolog_facts.append(f"{e1_type}({e1_name}).")
                prolog_facts.append(f"{e1_type}({raw_n1_name}).")
                # Process connected nodes and relationships
                connected = entity.get('connected_nodes') or []
                relations = entity.get('relations') or []

                for idx, node in enumerate(connected):
                    node_id = node.get('id')
                    raw_n2_name = node.get('name') if node.get('name') is not None else f"node_{node_id}"
                    e2_name = self._to_prolog_atom(raw_n2_name, fallback=f"node_{node_id}")

                    raw_e2_label = None
                    if node.get('type'):
                        t2 = node.get('type')
                        if isinstance(t2, (list, tuple)) and len(t2) > 0:
                            raw_e2_label = t2[0]
                        else:
                            raw_e2_label = t2
                    e2_type = self._label_to_prolog_predicate(raw_e2_label)
                    # prolog_facts.append(f"{e2_type}({e2_name}).")
                    prolog_facts.append(f"{e2_type}({raw_n2_name}).")

                    if idx < len(relations):
                        relation_raw = relations[idx]
                        relation_pred = self._label_to_prolog_predicate(relation_raw)
                        prolog_facts.append(f"{relation_pred}({raw_n1_name}, {raw_n2_name}).")

            except Exception as e:
                print(f"[WARN] Skipping malformed entity: {e}")
                continue

        prolog_facts = sorted(set(prolog_facts))


                    # 5. Verify requirement satisfaction conditions
                    # 6. Provide safety implications analysis
                    
                                    
                    # KNOWLEDGE BASE (DO NOT MODIFY):
                    # {chr(10).join(prolog_facts)}
        # prompt engineering
        input_text = f"""
                  ROLE: You are an expert in automotive safety engineering and autonomous vehicle systems. 
                  Your task is to analyze the safety requirements of an automated driving system (ADS) using the provided structured knowledge base.

            CONTEXT:
            The knowledge base is provided in structured data format as follows:
            {', '.join(prolog_facts)}
            
            INSTRUCTIONS:
            1. Identify and summarize the key safety requirements relevant to the input question.
            2. Analyze the knowledge base to extract all evidence and relations that support or describe these safety requirements.
            3. Evaluate whether the described system satisfies each requirement, citing explicit elements or facts from the knowledge base.
            4. DEPENDENCY TRACING: For every  requirement, list all directly referenced elements or entities that appear *exactly* in the knowledge base (no fabricated or inferred data) by their types.\
                For example, the traced elements include algorithm(Object Tracking),consist(Object Tracking, Adaptive Cruise Control)
                List all directly referenced single elements or entities (e.g., algorithm(Object Tracking)) that appear exactly in the knowledge base.
                exclude compound facts or relations (e.g., consist(Object Tracking, Adaptive Cruise Control),collect_data(Object Tracking, Mono Camera)), because they contain multiple elements.
            5. Provide a clear final answer that only includes the Dependency-traced elements within the knowledge base.
            6. Based on the retrieved  Dependency-traced elements, generate the rules between the input requirements and the traced elements following the prolog grammar by using the name of elements.\
                Do not change the name when generating the prolog code. For example Lidar should be 'Lidar'.
            7. For example, given a requirement  from the input prompts, we firstly give a nickname such as Req-A and then make it as fact Req(Req-A)
            then the schemas of the ruls should look like as follows: 
                requirement_model( Req-A Model Name).
                requirement_algorithm( Req-A Algorithm Name).
                rrequirement_sensor( Req-A, Sensor Name).
                
            To this end, the rulese should desribe the above relationships as follows:
                req_related_sensor(Req, S) :- requirement(Req), requirement_sensor(Req, S).

            OUTPUT FORMAT:
            - **Dependency Trace of listing all directly referenced single elements or entities (Exact Elements by their types)**       
            
            - ** The Prolog-based Rules **        
                    
                    USER QUESTION: {user_question}
                
                    Final ANSWER:
                    """

        # print("\n[OPTIMIZED LLM PROMPT]\n" + input_text)
        # print (keywords)
        return self.generate_with_llm(input_text)
    def generate_with_llm(self, input_text):
        """Generate response using Gemini LLM with output cleaning."""
        try:
            response = model.generate_content(input_text)
            return self.clean_response(response.text)
        except Exception as e:
            return "Sorry, I couldn't generate an answer due to an error."

    def clean_response(self, text):
        """Remove Markdown formatting and redundant symbols."""
        text = text.replace("*", "").replace("###", "").strip()
        return "\n".join([line.strip() for line in text.split("\n") if line.strip()])

    def rag_pipeline(self, user_question):
        """Execute the RAG pipeline with embedding-based retrieval and in-context learning."""
        relevant_entities = self.retrieve_relevant_entities(user_question)
        return self.generate_answer(user_question, relevant_entities)


def main():
    rag_system = Neo4jRAGSystem(
        uri=configuration.NEO4J_URI,
        user=configuration.NEO4J_USER,
        password=configuration.NEO4J_PASSWORD
    )
    try:
        print("Updating embeddings...")
        rag_system.update_embeddings()
        user_input = input("Enter your question: ")
        answer = rag_system.rag_pipeline(user_input)
        print("\nFinal Answer:", answer)
    finally:
        rag_system.close()


if __name__ == "__main__":
    main()