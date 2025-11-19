# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: Knowledge_safety

@File Name: main_rag_mlsafety.py
@Software: Python
@Time: Sep/2025
@Author: Rui Xu
@Contact: rxu@kth.se
@Version: 0.3.2
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
from function_call import SystemTools, FactGenerator
from google import genai as gen


# Configure Gemini API
genai.configure(api_key=configuration.GEMINI_API_KEY)
model = genai.GenerativeModel(configuration.GENERATION_MODEL)
# client = gen.Client(api_key=configuration.GEMINI_API_KEY)


class Neo4jRAGSystem:
    def __init__(self, uri, user, password, cache_file=configuration.CACHE_FILE):
        self.driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=60)
        self.cache_file = cache_file
        self.embedding_cache = self.load_cache()
        self.client = gen.Client(api_key=configuration.GEMINI_API_KEY_1)

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
                text = record['name']
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

    def extract_keywords(self, query_text, stop_words=configuration.STOP_WORDS, similarity_threshold=configuration.KEYWORD_SIMILARITY_THRESHOLD):
        tokens = re.findall(r'\b\w+\b', query_text)
        tokens = [t for t in tokens if
                  t.lower() not in stop_words]
        tokens = [t for t in tokens if len(t) >= 1]
        # add bigrams
        bigrams = [' '.join([tokens[i], tokens[i + 1]]) for i in range(len(tokens) - 1)]
        candidates = list(dict.fromkeys(tokens + bigrams))

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

        if combined_score < 0.2:
            depth = 1
        elif 0.2 <= combined_score < 0.4:
            depth = 2
        elif 0.4 <= combined_score < 0.6:
            depth = 3
        elif 0.6 <= combined_score < 0.8:
            depth = 4
        else:
            depth = 5

        if re.search(r'\b(trace|chain|path|flow|pipeline|scenario|environment|context)\b', query_text, re.I):
            depth = max(depth, 5)

        return depth

    def compute_syntactic_complexity(self, query_text):
        """
        Compute syntactic complexity based on sentence structure.
        """
        words = query_text.split()

        # Normalize sentence length to a max of 15 words.
        length_score = min(len(words) / 15, 1.0)

        wh_words = configuration.WH_WORDS
        wh_score = min(sum(1 for word in words if word.lower() in wh_words) / 2, 1.0)

        # Count clauses
        clause_count = len(re.findall(r',|;| but | however | although ', query_text))
        clause_score = min(clause_count / 2, 1.0)

        # Compute weighted complexity score.
        return (length_score * configuration.LENGTH_WEIGHT +
                wh_score * configuration.WH_WORD_WEIGHT +
                clause_score * configuration.CLAUSE_WEIGHT)

    def local_extract_domain_terms(self, req_text):
            """
            Local replica of extract_svo() from domain.py,
            but defined inside this function so no external import is needed.
            """
            prompt = """
            You are given a requirement sentence related to Autonomous Driving Systems.
            Extract domain-specific technical phrases.

            Rules:
            - Include only meaningful technical or domain-specific phrases.
            - Exclude generic functional words ("shall", "ensure", etc.).
            - Prefer noun phrases or short domain expressions.
            - Each phrase must appear exactly in the text.
            - No duplicates.
            - Ignore content inside brackets.
            - Output ONLY JSON array of strings.
            """

            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt + "\n\n" + req_text,
                    config={
                        "temperature": 0.0,
                        "response_mime_type": "application/json"
                    }
                )
                return [t.lower() for t in json.loads(response.text)]
            except Exception as e:
                print(f"[WARN] domain-term extraction failed: {e}")
                return []

    def retrieve_relevant_entities(self, query_text,
                                   top_k=configuration.TOP_K_RESULTS,
                                   similarity_threshold=configuration.SIMILARITY_THRESHOLD):
        """
        Hybrid retrieval that prioritizes keyword-driven embedding search.

        Steps:
        1. Extract domain terms via LLM (local_extract_domain_terms).
        2. Compute embeddings for each domain term (once).
        3. For each candidate DB node, compute the max cosine similarity between
           node.embedding and any domain-term embedding.
        4. If no domain-term embeddings are available, fall back to sentence embedding.
        5. Use max-term-sim as the primary filter; then combine with a lexical
           domain-term match score for reranking.
        """
        # 1) Domain term extraction (LLM)
        domain_terms = self.local_extract_domain_terms(query_text)
        if not domain_terms:
            # fallback to original keyword extractor
            domain_terms = self.extract_keywords(query_text)

        # 2) Compute embeddings for domain terms (cache results to avoid double calls)
        term_embeddings = []
        for term in domain_terms:
            emb = None
            try:
                emb = self.get_embedding(term)
            except Exception:
                emb = None
            if emb is not None:
                term_embeddings.append((term, emb))

        # 3) If we couldn't get any term embeddings, fall back to whole-sentence embedding
        use_query_emb = False
        query_emb = None
        if not term_embeddings:
            query_emb = self.get_embedding(query_text)
            if query_emb is None:
                print("[ERROR] Query embedding failed and no term embeddings available.")
                return []
            use_query_emb = True

        # 4) Determine depth/hop limit
        depth = self.determine_query_depth(query_text)
        hop_limit = min(depth, 4)
        neighbor_limit = 10

        # 5) Similarity retrieval using term embeddings as the reference
        sim_candidates = []
        seen_ids = set()

        with self.driver.session() as session:
            try:
                labels_res = session.run("CALL db.labels()")
                db_labels = {r["label"] for r in labels_res}
            except Exception:
                db_labels = set()

            target_labels = configuration.DESIRED_LABELS.intersection(db_labels)
            if not target_labels:
                target_labels = db_labels or configuration.DESIRED_LABELS

            for label in target_labels:
                try:
                    q = f"""
                    MATCH (n:{label})
                    WHERE n.embedding IS NOT NULL
                    RETURN elementId(n) AS id, n.name AS name,
                           n.embedding AS embedding, labels(n) AS labels
                    """
                    results = session.run(q)

                    for rec in results:
                        rid = rec["id"]
                        if rid in seen_ids:
                            continue

                        emb = rec["embedding"]
                        # normalize embedding if it's stored as JSON string
                        if isinstance(emb, str):
                            try:
                                emb = json.loads(emb)
                            except Exception:
                                continue
                        if emb is None:
                            continue

                        # compute similarity: max over term embeddings OR sentence embedding fallback
                        node_emb = emb
                        sim_score = 0.0
                        if use_query_emb:
                            # fallback: compare with entire sentence embedding
                            sim_score = self.cosine_similarity(query_emb, node_emb)
                        else:
                            # compute max similarity over domain-term embeddings
                            max_sim = 0.0
                            for _, term_emb in term_embeddings:
                                s = self.cosine_similarity(term_emb, node_emb)
                                if s > max_sim:
                                    max_sim = s
                            sim_score = max_sim

                        # filter by threshold
                        if sim_score >= similarity_threshold:
                            sim_candidates.append((rec, sim_score))
                            seen_ids.add(rid)

                except Exception as e:
                    print(f"[WARN] Similarity retrieval failed for label {label}: {e}")

        # sort and keep top_k by sim score
        sim_candidates.sort(key=lambda x: x[1], reverse=True)
        base_nodes = sim_candidates[:top_k]

        # 6) Depth-based neighbor expansion (BFS by hop)
        expanded = {}
        with self.driver.session() as session:
            for rec, sim_score in base_nodes:
                nid = rec["id"]
                expanded[nid] = {
                    "id": nid,
                    "name": rec.get("name") or "",
                    "labels": rec.get("labels") or [],
                    "embedding": rec.get("embedding"),
                    "sim": sim_score
                }

                for hop in range(1, hop_limit + 1):
                    try:
                        q = f"""
                        MATCH (n) WHERE elementId(n)=$nid
                        MATCH (n)-[*{hop}]-(m)
                        RETURN DISTINCT elementId(m) AS id,
                                        m.name AS name,
                                        labels(m) AS labels,
                                        m.embedding AS embedding
                        LIMIT {neighbor_limit}
                        """
                        neighbors = session.run(q, {"nid": nid}).data()
                        for nb in neighbors:
                            nbid = nb["id"]
                            if nbid not in expanded:
                                expanded[nbid] = {
                                    "id": nbid,
                                    "name": nb.get("name") or "",
                                    "labels": nb.get("labels") or [],
                                    "embedding": nb.get("embedding"),
                                    "sim": 0.0
                                }
                    except Exception:
                        continue

        # 7) Score final candidates: combine sim (from term-based retrieval) with lexical domain-term match
        scored = []
        domain_set = set(domain_terms)
        for nid, info in expanded.items():
            name = (info["name"] or "").lower()
            # lexical match score: how many domain terms appear in the node name
            kw_score = sum(1 for t in domain_set if t in name)
            kw_score = min(kw_score, 3) / 3.0  # normalize to [0,1]

            # use the sim already stored; if missing recalc as fallback
            sim_val = info.get("sim", 0.0)
            if sim_val == 0.0:
                # try compute with term embeddings if available
                node_emb = info.get("embedding")
                if isinstance(node_emb, str):
                    try:
                        node_emb = json.loads(node_emb)
                    except Exception:
                        node_emb = None
                if node_emb is not None and term_embeddings:
                    max_sim = 0.0
                    for _, term_emb in term_embeddings:
                        s = self.cosine_similarity(term_emb, node_emb)
                        if s > max_sim:
                            max_sim = s
                    sim_val = max_sim
                elif node_emb is not None and use_query_emb and query_emb is not None:
                    sim_val = self.cosine_similarity(query_emb, node_emb)

            # final score: weighted combination (adjust weights as needed)
            final_score = 0.6 * sim_val + 0.4 * kw_score
            scored.append((info, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        final_nodes = scored[:top_k]

        # 8) Normalize output schema (same shape as original)
        results = []
        with self.driver.session() as session:
            for info, score in final_nodes:
                nid = info["id"]
                try:
                    q = session.run(
                        """
                        MATCH (n) WHERE elementId(n)=$id
                        OPTIONAL MATCH (n)-[r]-(m)
                        RETURN collect({id:elementId(m), name:m.name, type:labels(m)}) AS cn,
                               collect(type(r)) AS rel,
                               labels(n) AS labels,
                               n.name AS name
                        """,
                        {"id": nid}
                    ).single()

                    raw = {
                        "node1": {"id": nid, "name": q["name"], "type": q["labels"]},
                        "relations": q["rel"],
                        "connected_nodes": q["cn"],
                        "source": "hybrid"
                    }
                except Exception:
                    raw = {
                        "node1": {"id": nid, "name": info["name"], "type": info["labels"]},
                        "relations": [],
                        "connected_nodes": [],
                        "source": "hybrid"
                    }

                results.append(self.normalize_entity(raw))

        print(f"[INFO] Hybrid retrieval returned {len(results)} entities.")
        return results

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

    def _safe_raw(self, text):
        """
        Keep raw text from DB, but wrap in quotes if it contains spaces or special characters,
        so that Prolog can still parse it correctly.
        """
        if text is None:
            return "unknown"
        text = str(text)
        if re.match(r'^[a-zA-Z0-9_]+$', text):
            return text
        return f"'{text}'"

    def _label_to_prolog_predicate(self, label):
        """
        Convert a graph label or relation name into a safe Prolog predicate name.
        """
        if not label:
            return "unknown"
        s = str(label).strip().lower().replace(" ", "_")
        s = re.sub(r'[^a-z0-9_]', '_', s)
        if not re.match(r'^[a-z_]\w*$', s):
            return f"'{label}'"
        return s

    def call_llm_for_structured_facts(self, user_question, current_prolog_facts):
        """
        Use function-calling style: ask LLM to return structured JSON describing
        new components/requirements/rules to be added.
        """
        schema_instruction = """
        Return JSON with up to three keys: components, requirements, rules.
        components: list of {name, sensors[], algorithms[], actuators[]}
        requirements: list of {req_id, target, sensors[], algorithms[], models[]}
        rules: list of {head, body[]} where head and body contain identifiers, DO NOT emit raw prolog punctuation.
        """

        prompt = f"""
        CURRENT_PROLOG_FACTS:
        {current_prolog_facts}

        USER_QUESTION:
        {user_question}

        INSTRUCTIONS:
        {schema_instruction}

        Output only JSON.
        """

        try:
            response = model.generate_content(prompt)
            raw = response.text
            parsed = json.loads(raw)
        except Exception as e:
            print(f"[WARN] LLM structured call failed: {e}")
            return None

        # Basic validation
        if not isinstance(parsed, dict):
            print("[WARN] LLM did not return JSON object.")
            return None

        # Use SystemTools to convert to facts
        tools = SystemTools(FactGenerator())

        new_facts = []
        for comp in parsed.get("components", []):
            try:
                new_facts += tools.add_new_component(
                    name=comp.get("name", "unknown"),
                    sensors=comp.get("sensors", []),
                    algorithms=comp.get("algorithms", []),
                    actuators=comp.get("actuators", [])
                )
            except Exception as e:
                print(f"[WARN] bad component entry: {comp} -> {e}")

        for req in parsed.get("requirements", []):
            try:
                new_facts += tools.add_new_requirement(
                    req_id=req.get("req_id", "REQ_UNKNOWN"),
                    target=req.get("target", "unknown"),
                    sensors=req.get("sensors", []),
                    algorithms=req.get("algorithms", []),
                    models=req.get("models", [])
                )
            except Exception as e:
                print(f"[WARN] bad requirement entry: {req} -> {e}")

        # rules can be handled by templating too, if present
        for r in parsed.get("rules", []):
            head = r.get("head")
            body = r.get("body", [])
            if head and isinstance(body, list):
                body_str = ", ".join(body)
                new_facts.append(f"{head} :- {body_str}.")

        return new_facts

    def generate_answer(self, user_question, relevant_entities):
        """
        Generate an answer using Gemini LLM with optimized prompt engineering.
        Single-function version (no nested helper functions).
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
                e1_name = self._safe_raw(raw_n1_name)

                # Extract primary label
                raw_e1_label = None
                if node1.get('type'):
                    t = node1.get('type')
                    if isinstance(t, (list, tuple)) and len(t) > 0:
                        raw_e1_label = t[0]
                    else:
                        raw_e1_label = t
                e1_type = self._label_to_prolog_predicate(raw_e1_label)
                prolog_facts.append(f"{e1_type}({e1_name}).")

                # Process connected nodes and relationships
                connected = entity.get('connected_nodes') or []
                relations = entity.get('relations') or []

                for idx, node in enumerate(connected):
                    node_id = node.get('id')
                    raw_n2_name = node.get('name') if node.get('name') is not None else f"node_{node_id}"
                    e2_name = self._safe_raw(raw_n2_name)

                    raw_e2_label = None
                    if node.get('type'):
                        t2 = node.get('type')
                        if isinstance(t2, (list, tuple)) and len(t2) > 0:
                            raw_e2_label = t2[0]
                        else:
                            raw_e2_label = t2
                    e2_type = self._label_to_prolog_predicate(raw_e2_label)
                    prolog_facts.append(f"{e2_type}({e2_name}).")

                    if idx < len(relations):
                        relation_raw = relations[idx]
                        relation_pred = self._label_to_prolog_predicate(relation_raw)
                        prolog_facts.append(f"{relation_pred}({e1_name}, {e2_name}).")

            except Exception as e:
                print(f"[WARN] Skipping malformed entity: {e}")
                continue

        # canonicalize DB facts
        prolog_facts = sorted(set(prolog_facts))
        facts_str = "\n".join(prolog_facts) if prolog_facts else "% No facts available"

        # Call function-calling wrapper to request structured facts from the LLM
        llm_generated_facts = None
        try:
            llm_generated_facts = self.call_llm_for_structured_facts(user_question, facts_str)
        except Exception as e:
            print(f"[WARN] call_llm_for_structured_facts error: {e}")
            llm_generated_facts = None

        # Normalize LLM output to candidate lines
        cand_lines = []
        if llm_generated_facts:
            if isinstance(llm_generated_facts, str):
                cand_lines = [ln.strip() for ln in llm_generated_facts.splitlines() if ln.strip()]
            elif isinstance(llm_generated_facts, dict):
                found = False
                for k in ("facts", "prolog", "lines", "generated_facts"):
                    if k in llm_generated_facts:
                        val = llm_generated_facts[k]
                        if isinstance(val, list):
                            cand_lines.extend([str(x).strip() for x in val if x])
                        elif isinstance(val, str):
                            cand_lines.extend([ln.strip() for ln in val.splitlines() if ln.strip()])
                        found = True
                        break
                if not found:
                    for v in llm_generated_facts.values():
                        if isinstance(v, str):
                            cand_lines.extend([ln.strip() for ln in v.splitlines() if ln.strip()])
                        elif isinstance(v, list):
                            cand_lines.extend([str(x).strip() for x in v if isinstance(x, (str, int))])
            elif isinstance(llm_generated_facts, (list, tuple)):
                for el in llm_generated_facts:
                    if isinstance(el, str):
                        cand_lines.append(el.strip())
                    else:
                        cand_lines.append(str(el).strip())

        # Validate candidate lines conservatively
        validated_llm_facts = []
        for ln in cand_lines:
            if not ln:
                continue
            if '\n' in ln or '\r' in ln:
                print(f"[WARN] Rejecting LLM fact (multiline): {ln}")
                continue
            if len(ln) > 2000:
                print(f"[WARN] Rejecting LLM fact (too long): {ln}")
                continue
            control_found = False
            for ch in ln:
                if ord(ch) < 9:
                    control_found = True
                    break
            if control_found:
                print(f"[WARN] Rejecting LLM fact (control char): {ln}")
                continue
            # basic syntactic shape: must contain '('
            if '(' not in ln:
                print(f"[WARN] Rejecting LLM fact (no '('): {ln}")
                continue
            if not ln.endswith('.'):
                ln = ln + '.'
            validated_llm_facts.append(ln)

        # Merge validated LLM facts into the prolog_facts
        if validated_llm_facts:
            merged = set(prolog_facts)
            for f in validated_llm_facts:
                if f not in merged:
                    merged.add(f)
            prolog_facts = sorted(merged)
        else:
            if llm_generated_facts is None:
                print(
                    "[INFO] LLM structured-facts call returned None or raised an error; proceeding with DB facts only.")
            else:
                print("[INFO] LLM returned no valid facts after validation; proceeding with DB facts only.")

        facts_str = "\n".join(prolog_facts) if prolog_facts else "% No facts available"

        # Prompt engineering
        # input_text = f"""
        # ROLE:
        # You are an expert in automotive safety engineering and autonomous driving systems.
        # Your task is to analyze the safety requirements of an Automated Driving System (ADS)
        # using the provided structured knowledge base written in Prolog-like format.
        #
        # KNOWLEDGE BASE (DO NOT MODIFY):
        # {facts_str}
        #
        # USER QUESTION:
        # {user_question}
        #
        # INSTRUCTIONS:
        # 1. Identify and summarize the key safety requirements relevant to the question.
        # 2. Analyze the knowledge base to find all directly referenced single-element entities and facts that describe or support these requirements.
        # 3. Evaluate whether the described system satisfies each requirement, citing exact elements from the knowledge base.
        # 4. Perform DEPENDENCY TRACING:
        #    - List all directly referenced single-argument facts (e.g., algorithm(ObjectTracking)., sensor(Lidar).).
        #    - Exclude compound relations (e.g., consist(ObjectTracking, ACC). or serve(Mono Camera, Object Tracking).).
        #    - Each fact must appear exactly as written in the knowledge base, preserving capitalization, spacing, and formatting.
        # 5. Provide a final structured answer with two strictly separated sections:
        #    (1) Dependency Trace: all matched single-argument entities.
        #    (2) Prolog-Based Rules: relations connecting these entities to requirements.
        # 6. Generate Prolog rules that define relationships between each requirement (ReqA) and its matched entities.
        #    - Example schema:
        #         requirement_algorithm(ReqA, AlgorithmName).
        #         requirement_sensor(ReqA, SensorName).
        #         requirement_model(ReqA, ModelName).
        #         requirement_component(ReqA, ComponentName).
        #         requirement_system_description(ReqA, SystemDescriptionName).
        #    - Example rule definitions:
        #         reqrelated_algorithm(ReqA, Algo) :- requirement(ReqA), requirement_algorithm(ReqA, Algo).
        #         reqrelated_sensor(ReqA, S) :- requirement(ReqA), requirement_sensor(ReqA, S).
        #         reqrelated_model(ReqA, M) :- requirement(ReqA), requirement_model(ReqA, M).
        #         reqrelated_component(ReqA, C) :- requirement(ReqA), requirement_component(ReqA, C).
        #         reqrelated_system_description(ReqA, SD) :- requirement(ReqA), requirement_system_description(ReqA, SD).
        #
        # ENTITY NAME INTEGRITY RULES (CRITICAL):
        # - You MUST preserve exact spelling, capitalization, and spacing of all entity names as they appear in the knowledge base.
        # - Never modify, quote, or translate any entity name.
        # - Treat entity names as literal identifiers.
        # - Any quotation marks or punctuation around entity names will be removed automatically after generation, so output them as-is.
        #
        # IMPORTANT CONSTRAINTS:
        # - Do not invent or infer missing entities.
        # - If no matching elements are found, explicitly write: "No matching element found."
        # - Use only minimal and clear text — do not output natural-language reasoning.
        # - The final answer must be fully structured, containing only the two required sections.
        #
        # OUTPUT FORMAT (STRICT):
        # Dependency Trace
        # [list of single-argument entities, one per line, ending with '.']
        #
        # Prolog-Based Rules
        # [requirement_* and reqrelated* facts, one per line, ending with '.']
        #
        # Final Answer:
        # """
        input_text = f"""
        ROLE:
        You are an expert in automotive safety engineering and autonomous driving systems.
        Your task is to analyze the safety requirements of an Automated Driving System (ADS)
        using the provided structured knowledge base written in Prolog-like format.

        KNOWLEDGE BASE (DO NOT MODIFY):
        {facts_str}

        USER QUESTION:
        {user_question}

        INSTRUCTIONS:
        1. Identify and summarize the key safety requirements relevant to the question.
        2. Analyze the knowledge base to find all directly referenced single-element entities and facts that describe or support these requirements.
        3. Evaluate whether the described system satisfies each requirement, citing exact elements from the knowledge base.
        4. Perform DEPENDENCY TRACING:
           - List all directly referenced single-argument facts (e.g., algorithm(ObjectTracking)., model(YOLOv5)., sensor(Lidar).).
           - Each fact must appear exactly as written in the knowledge base, preserving capitalization, spacing, and formatting.
        5. Provide a final structured answer with two strictly separated sections:
           Dependency Trace: all matched single-argument entities.

        ENTITY NAME INTEGRITY RULES (CRITICAL):
        - You MUST preserve exact spelling, capitalization, and spacing of all entity names as they appear in the knowledge base.
        - Never modify, quote, or translate any entity name.
        - Treat entity names as literal identifiers.
        - Any quotation marks or punctuation around entity names will be removed automatically after generation, so output them as-is.

        IMPORTANT CONSTRAINTS:
        - Do not invent or infer missing entities.
        - If no matching elements are found, explicitly write: "No matching element found."
        - Use only minimal and clear text — do not output natural-language reasoning.
        - The final answer must be fully structured, containing only the two required sections.

        OUTPUT FORMAT (STRICT):
        Dependency Trace
        [list of single-argument entities, one per line, ending with '.']

        """

        print("\n[OPTIMIZED LLM PROMPT]\n" + input_text)
        return self.generate_with_llm(input_text)

    def generate_with_llm(self, input_text):
        """Generate response using Gemini LLM with output cleaning."""
        try:
            response = model.generate_content(input_text)
            return self.clean_response(response.text)
        except Exception as e:
            return "Sorry, I couldn't generate an answer due to an error."

    def normalize_entity(self, raw_entity):
        """
        Normalize a raw retrieval record into a stable schema
        """
        if not raw_entity:
            return {"node1": {"id": None, "name": "", "type": []},
                    "relations": [], "connected_nodes": [], "source": None}

        node1 = raw_entity.get("node1") or {}
        nid = node1.get("id")
        nname = node1.get("name") or ""
        ntype = node1.get("type") or []
        # ensure type is list
        if isinstance(ntype, (str,)):
            ntype = [ntype]
        elif not isinstance(ntype, (list, tuple)):
            ntype = list(ntype) if ntype is not None else []
        else:
            ntype = list(ntype)

        # connected nodes normalization
        raw_conns = raw_entity.get("connected_nodes") or []
        conns = []
        for cn in raw_conns:
            if not cn:
                continue
            cid = cn.get("id")
            cname = cn.get("name") or ""
            ctype = cn.get("type") or []
            if isinstance(ctype, (str,)):
                ctype = [ctype]
            elif not isinstance(ctype, (list, tuple)):
                ctype = list(ctype) if ctype is not None else []
            else:
                ctype = list(ctype)
            conns.append({"id": cid, "name": cname, "type": ctype})

        relations = raw_entity.get("relations") or []
        if isinstance(relations, (str,)):
            relations = [relations]
        elif not isinstance(relations, (list, tuple)):
            try:
                relations = list(relations)
            except Exception:
                relations = []

        return {
            "node1": {"id": nid, "name": nname, "type": ntype},
            "relations": list(relations),
            "connected_nodes": conns,
            "source": raw_entity.get("source")
        }

    def clean_response(self, text: str) -> str:
        if not text:
            return ""

        s = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = [ln.strip() for ln in s.split('\n') if ln.strip()]

        deps, facts, rules = [], [], []

        dep_pred = re.compile(
            r'^(algorithm|model|sensors?|sensor|actuator|actuators?|component|dataset|system_description)\s*\(.*\)\.?$',
            re.IGNORECASE
        )
        fact_pred = re.compile(r'^requirement_[A-Za-z0-9_]*\s*\(.*\)\.?$', re.IGNORECASE)
        rule_pred = re.compile(r'.*:-.*')
        reqrel_pred = re.compile(r'^reqrelated_[A-Za-z0-9_]*', re.IGNORECASE)

        for ln in lines:
            low = ln.lower()
            if low in ("final answer:", "final answer", "prolog-based facts", "prolog-based rules", "dependency trace"):
                continue

            if rule_pred.search(ln) or reqrel_pred.match(ln):
                rules.append(ln if ln.endswith('.') else ln + '.')
                continue
            if fact_pred.match(ln):
                facts.append(ln if ln.endswith('.') else ln + '.')
                continue
            if dep_pred.match(ln):
                deps.append(ln if ln.endswith('.') else ln + '.')
                continue
            if '(' in ln and ')' in ln:
                deps.append(ln if ln.endswith('.') else ln + '.')
                continue

        def quote_after_comma(line):
            # split at first '('
            if '(' not in line or ')' not in line:
                return line
            head, rest = line.split('(', 1)
            args = rest.rstrip(').').split(',')
            new_args = []
            for a in args:
                core = a.strip().strip("'\"")
                new_args.append(f"'{core}'")
            return f"{head}({', '.join(new_args)})."

        facts = [quote_after_comma(f) for f in facts]
        deps = [quote_after_comma(d) for d in deps]

        out = []
        out.append(" ")
        out.append("Dependency Trace")
        out.extend(deps)
        out.append("")

        # if facts:
        #     out.append("Prolog-Based Facts")
        #     out.extend(facts)
        #     out.append("")
        #
        # if rules:
        #     out.append("Prolog-Based Rules")
        #     out.extend(rules)

        return "\n".join(out)

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
