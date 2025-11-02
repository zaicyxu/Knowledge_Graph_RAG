# Input Queries

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

    The **vehicle control module shall predict and maintain a feasible trajectory** within current lane boundaries using current and

11，**Object Detection Accuracy**

​	The object detection module shall achieve at least 95% precision and recall for pedestrian detection 	under nominal lighting conditions.

12. **Model Update Mechanism**

    The ML model deployment process shall support over-the-air updates for trajectory prediction models without system downtime.

13. **Sensor Redundancy**

    The system shall maintain functional safety through redundant sensor inputs from both camera and lidar for critical object detection tasks.

14. **Real-Time Processing**

    All perception algorithms shall process sensor data within 100 milliseconds to support real-time decision-making.

15. **Brake Actuation Latencyv**

    The PAEB system shall activate brakes within 150 milliseconds after pedestrian detection confirmation.

16. **Trajectory Prediction Confidence**

    The trajectory prediction module shall output a confidence score for each predicted path, with a minimum threshold of 80% for actuation.

17. **Adaptive Cruise Control Smoothness**

    The ACC system shall adjust throttle and brake inputs smoothly to maintain passenger comfort while following a leading vehicle.

18. **Lane Keeping Assist Boundaries**

    The lane keeping system shall only activate when lane boundaries are detected with high confidence and vehicle speed exceeds 60 km/h.

19. **ML Safety Assurance Traceability**

    Each ML safety requirement shall be traceable to one or more system-level safety requirements through the ML development flow.

20. **End-to-End Latency Budget**

    The total latency from sensor data capture to actuator command shall not exceed 200 milliseconds for any safety-critical function.