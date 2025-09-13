# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
@File Name: configuration.py
@Description: Store sensitive information like API keys and database credentials
"""

# Gemini API Key
GEMINI_API_KEY = "AIzaSyBdTPqt4RpQvOc676Z1v_OuEkDsqhrJd9k"

# Neo4j Database Credentials
NEO4J_URI = "bolt://localhost:7687"
NEO4J_BLOOM_URI = "http://localhost:7474/bloom"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

# Key entity types and relationships
KEY_ENTITIES = {
    "manufacturer": ["Work_on", "Certify"],
    "material": ["Process"],
    "industry": ["Belong", "Sub_Industry"],
    "certification": ["Certify"],
    "energy": ["Power_from"]
}

RELATIONSHIP_MAPPING = {
    "Belong": "Industry",
    "Certify": "Certification",
    "Process": "Material",
    "Work_on": "Service",
    "Sub_Business": "Service",
    "Sub_Industry": "Industry",
    "Power_from": "Power"
}