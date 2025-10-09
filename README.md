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

1. Example of the input questions. (Requirements)
   1. Object tracks shall be updated within 100 ms after receiving new sensor data.
   2. The object detector shall reliably detect pedestrians of ≥ 0.4 m width at a distance of up to 40 m with a probability of ≥ 0.9.
   3. The segmentation network shall identify lane boundaries with a lateral error ≤ 0.2 m at 30 m range under nominal lighting.
   4. The ML perception module shall periodically self-validate using reference scenes or known calibration targets to ensure runtime performance consistency.
   5. Measure time stamps on frame acquisition and on completed inference; compute median and 95th percentile over a representative trace (≥10,000 frames). Pass if med ≤ 30 ms and p95 ≤ 60 ms.
   6. The object detector shall detect pedestrians of physical width ≥ 0.40 m at distances up to 40 m with true positive probability P ≥ 0.90 under nominal daytime lighting and unobstructed view (IoU threshold = 0.5).
   7. The segmentation / lane detection network shall localize lane boundaries with lateral error ≤ 0.20 m at 30 m range under nominal lighting; RMS lateral error ≤ 0.12 m across the validated dataset.
   8. During Hardware-in-the-Loop tests, end-to-end latency from sensor timestamp (t_sensor) to simulated actuation update (t_actuate) shall exhibit median ≤ 80 ms and 95th-percentile ≤ 180 ms. Measured latency must include serialization, network/IPC delay, inference, and simulation input processing.
