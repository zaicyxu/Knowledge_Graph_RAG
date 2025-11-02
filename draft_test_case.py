#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 22:10:09 2025

@author: pengsu_workstation
"""

        5. Provide a final list containing **only** the dependency-traced elements that appear
           exactly in the knowledge base.
        
        6. PROLOG RULE GENERATION:
           - Treat each input question as exactly one requirement.
           - Assign it a unique label (e.g., Req-A) and declare it as:
                 requirement(Req-A).
           - For every traced element, generate a rule connecting the requirement to the element
             using the following schemas:
                 requirement_model(Req-A, ModelName).
                 requirement_algorithm(Req-A, AlgorithmName).
                 requirement_sensor(Req-A, SensorName).
           - **Do not rename or infer any element names.** Use them exactly as found in the
             dependency-traced list (e.g., keep 'Lidar' as 'Lidar').
        
        7. Define relationship rules describing these dependencies, for example:
               req_related_sensor(Req, S) :- requirement(Req), requirement_sensor(Req, S).
        
        8. FINAL OUTPUT FORMAT:
           The final response must contain exactly **two sections** and nothing else:
        
           **Dependency Trace Elements:**
           <list of exact dependency-traced elements>
        
           **Prolog-Based Rules:**
           <generated Prolog rules>