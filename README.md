# Traceability Analysis of LLM and Prolog

## Current Issues

#### ~~Issues from the LLM Sides~~

1. ~~The Final Answer from LLM is not stable yet. For example, regarding the following prompts, the LLM sometimes make final answers  somtimes nor.~~

	- ~~Which kind of algorithm has been used in Autonomous braking system?~~
	- ~~What kind of functionalities containing this system?~~
	- ~~Which sensor is most related to the pedestrian detection?~~

#### Issues from the Orchestrator Sides

1. Unify the name between LLM and Prolog.

​	e.g., mono_camera and Mono Camera.

2. What's the queries for Prolog ???

#### Issues from Current Framework

1. Whats' the relationships between Neo4j and Prolog?

a) Prolog generates the inferred graphs, which contain the facts existed in the Logic knowledgebase.

2. What's the input of the framework?

a) Test cases and other system-related questions

3. What's the main workflow of the framework?

a) **Constructing the graph-based knowledgebase**

b) **Constructing the logic-based knowledgebase**

​	b.1) Input: Requirement with Nature Language Representations

​	b.2) LLM-based Interpreter used to translate the reuqirements to logic representations (RAG).

​	b.3) LLM-based aligner used to support the multiple step reasoning via Prolog.

​	b.4) The purpose of logic-based knowledgebase is to map the requirements with the design-time information.

​	**Key Issues**:

​	

 c) **Traceability analysis regarding the design information.**

​	The design information includes: 1) Test cases 2) Data argument statement....

​	c.1) Input: Test cases and other systems-related questions

​	c.2) LLM-based Interprete

​	c.3) Prolog-based logic solver.



​	**Some examples of test cases:**

​		**TC-P01 / Cut-in detection (highway)**

- **Env:** Proving ground, dry daylight.
- **Stimuli:** Lead car 100 km/h; adjacent car cuts in with Δv=-10 km/h, gap 15 m.
- **Expected:** ADS re-tracks new lead in ≤300 ms; no false emergency brake.
- **Metrics:** Track switch latency ≤300 ms; lateral jerk ≤3 m/s²; min TTC ≥2.0 s.

​	

​	**TC-P02 / Motorcycle in blind spot**

- **Env**: Sim + track.

- **Stimuli:** Motorcycle approaches from rear quarter at 120 km/h.

- **Expected:** Correct classification in blind-spot; inhibit lane change.

- **Metrics:** Detection range ≥50 m; false lane-change probability = 0.

  

​	**TC-P03 / Occluded pedestrian at mid-block**

- **Env:** Urban mock street; parked van occludes 80% of body.
- **Stimuli:** Pedestrian emerges walking 1.2 m/s.
- **Expected:** Early hazard hypothesis; speed reduction before full visibility.
- **Metrics:** Speed ≤15 km/h before full reveal; no collision; min clearance ≥1.5 m.
- 

 d) **Whats the purpose of Prolog?**





## Updated on 7th Oct


Example of the input questions. (Requirements)

1. **Object Tracking Update**

   The system **shall update all object tracks upon receipt of new sensor data** to ensure accurate and consistent environmental representation.

2. **Pedestrian Detection**

   The **object detection module shall reliably detect pedestrians** with accurate width and position estimation at a minimum distance of *[specify distance, e.g., 50 m]* under nominal conditions.

3. **Lane Boundary Segmentation**

   The **semantic segmentation network shall identify lane boundaries** and drivable areas under nominal lighting and weather conditions.

4. **Perception Self-Validation**

   The **machine learning (ML) perception module shall periodically self-validate** using reference scenes or calibration targets to ensure runtime consistency and prevent model drift.

5. **Lidar Semantic Segmentation**

   The **lidar perception subsystem shall perform semantic segmentation** of point cloud data and distribute classified data to functional components such as obstacle detection and free-space estimation.

6. **Sensor Fusion**

   The **sensor fusion module shall integrate data from lidar, radar, camera, and ultrasonic sensors** to generate a unified and temporally consistent environmental model.

7. **Environmental Awareness**

   Each **sensor subsystem shall analyze its respective portion of the external environment** (e.g., radar for long-range objects, cameras for visual classification, lidar for depth and geometry).

8. **Parking Assistance**

   The **parking assistance module shall detect and evaluate available parking spaces**, determine suitability, and guide the vehicle during parking maneuvers.

9. **Adaptive Cruise and Car-Following**

   The **adaptive cruise control and car-following modules shall detect leading vehicles and lane markings** to maintain safe following distances and lane discipline.

10. **Lane Keeping and Trajectory Prediction**

    The **vehicle control module shall predict and maintain a feasible trajectory** within current lane boundaries using current and anticipated environment data.
