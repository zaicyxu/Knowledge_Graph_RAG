# Input Queries

1. The system **shall update all object tracks upon receipt of new sensor data** to ensure accurate and consistent environmental representation.

2. The **object detection module shall reliably detect pedestrians** with accurate width and position estimation at a minimum distance of *[specify distance, e.g., 50 m]* under nominal conditions.
3. The **semantic segmentation network shall identify lane boundaries** and drivable areas under nominal lighting and weather conditions.
4. The **machine learning (ML) perception module shall periodically self-validate** using reference scenes or calibration targets to ensure runtime consistency and prevent model drift.
5. The **lidar perception subsystem shall perform semantic segmentation** of point cloud data and distribute classified data to functional components such as obstacle detection and free-space estimation.
6. The **sensor fusion module shall integrate data from lidar, radar, camera, and ultrasonic sensors** to generate a unified and temporally consistent environmental model.
7. Each **sensor subsystem shall analyze its respective portion of the external environment** (e.g., radar for long-range objects, cameras for visual classification, lidar for depth and geometry).
8. The **parking assistance module shall detect and evaluate available parking spaces**, determine suitability, and guide the vehicle during parking maneuvers.
9. The **adaptive cruise control and car-following modules shall detect leading vehicles and lane markings** to maintain safe following distances and lane discipline.
10. The **vehicle control module shall predict and maintain a feasible trajectory** within current lane boundaries
11. The object detection module shall achieve at least 95% precision and recall for pedestrian detection 	under nominal lighting conditions.
12. The ML model deployment process shall support over-the-air updates for trajectory prediction models without system downtime.
13. The system shall maintain functional safety through redundant sensor inputs from both camera and lidar for critical object detection tasks.
14. All perception algorithms shall process sensor data within 100 milliseconds to support real-time decision-making.
15. The PAEB system shall activate brakes within 150 milliseconds after pedestrian detection confirmation.
16. The trajectory prediction module shall output a confidence score for each predicted path, with a minimum threshold of 80% for actuation.
17. The ACC system shall adjust throttle and brake inputs smoothly to maintain passenger comfort while following a leading vehicle.
18. The lane keeping system shall only activate when lane boundaries are detected with high confidence and vehicle speed exceeds 60 km/h.
19. Each ML safety requirement shall be traceable to one or more system-level safety requirements through the ML development flow.
20. The total latency from sensor data capture to actuator command shall not exceed 200 milliseconds for any safety-critical function.