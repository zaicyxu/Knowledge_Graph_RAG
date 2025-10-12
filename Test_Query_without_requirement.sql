
// Define the system requirement .....

// Define the system description .....
Merge (sys_des:System_Description {
  name: "System_Description",
  type: "Pyhsical View"
})



// Define the hardware.....
Merge (radar:Sensors {
  name: "Radar",
  type: "Physical View"
})

Merge (Lidar:Sensors {
  name: "Lidar",
  type: "Physical View"
})

Merge (camera:Sensors {
  name: "Mono Camera",
  type: "Physical View"
})

Merge (brakes:actuators {
  name: "Brakes",
  type: "Physical View"
})

// Define main functional descriptions.....

Merge (AEB:Component {
  name: "PAEB",
  type: "Functional View"
})

Merge (LaneKeeping:Component {
  name: "Lane Keeping",
  type: "Functional View"
})

Merge (ACC:Component {
  name: "Adaptive Cruise Control",
  type: "Functional View"
})

Merge (RemoteParking:Component {
  name: "Automated Parking Assist",
  type: "Functional View"
})

// Define main ML blocks descriptions.....

Merge (LaneDetection:Algorithm {
  name: "Lane Detection",
  type: "Functional View"
})

Merge (SpaceDetection:Algorithm {
  name: "Space Detection",
  type: "Functional View"
})

Merge (ObjectDetection:Algorithm {
  name: "Object Detection",
  type: "Functional View"
})
Merge (ObjectTracking:Algorithm {
  name: "Object Tracking",
  type: "Functional View"
})
Merge (SemanticSegmantiation:Algorithm {
  name: "Semantic Segmantiation",
  type: "Functional View"
})
Merge (TrajectoryPrediction:Algorithm {
  name: "Trajectory Prediction",
  type: "Functional View"
})


// Define the Dataset Specification
Merge (Vehicles:Dataset {
  name: "Vehicles",
  type: "ODD Element"
})

Merge (Pedestrian:Dataset {
  name: "Pedestrian",
  type: "ODD Element"
})

Merge (LaneMark:Dataset {
  name: "Lane Mark",
  type: "ODD Element"
})

Merge (TrafficSign:Dataset {
  name: "Traffic Sign",
  type: "ODD Element"
})


Merge (Sunny:Dataset {
  name: "Sunny",
  type: "ODD Element"
})

Merge (Night:Dataset {
  name: "Night",
  type: "ODD Element"
})

Merge (Dawn:Dataset {
  name: "Dawn",
  type: "ODD Element"
})

Merge (CityRoad:Dataset {
  name: "City Road",
  type: "ODD Element"
})

Merge (Highway:Dataset {
  name: "Highway",
  type: "ODD Element"
})

// Define the ML description.....

    // Object Detection
Merge (YOLO:Model {
  name: "YOLOv5",
  type: "ML Component"
})
    // Object Tracking
Merge (TransTrack:Model {
  name: "TransTrack",
  type: "ML Component"
})

    // Trajectory Prediction 
Merge (Social:Model {
  name: "Social-LSTM",
  type: "ML Component"
})

    // Semantic Segmantiation 
Merge (PointNet:Model {
  name: "PointNet++",
  type: "ML Component"
})

    // Lane Tracking
Merge (ENet:Model {
  name: "ENet-SAD",
  type: "ML Component"
})


    // Space Detection
Merge (UNet:Model {
  name: "U-Net",
  type: "ML Component"
})


// System Specifications
MERGE (camera)-[:Consist]->(sys_des)
MERGE (radar)-[:Consist]->(sys_des)
MERGE (Lidar)-[:Consist]->(sys_des)
MERGE (brakes)-[:Consist]->(sys_des)
MERGE (AEB)-[:Consist]->(sys_des)
MERGE (LaneKeeping)-[:Consist]->(sys_des)
MERGE (ACC)-[:Consist]->(sys_des)
MERGE (RemoteParking)-[:Consist]->(sys_des)


// ML Specification

MERGE (ObjectDetection)-[:Include]->(AEB)
MERGE (SemanticSegmantiation)-[:Include]->(AEB)

MERGE (LaneDetection)-[:Include]->(ACC)
MERGE (ObjectTracking)-[:Include]->(ACC)


MERGE (TrajectoryPrediction)-[:Include]->(LaneKeeping)
MERGE (LaneDetection)-[:Include]->(LaneKeeping)


MERGE (SpaceDetection)-[:Include]->(RemoteParking)



// Dataset Requirement
MERGE (Sunny)-[:DataRequirement]->(AEB)
MERGE (Night)-[:DataRequirement]->(AEB)
MERGE (Dawn)-[:DataRequirement]->(AEB)
MERGE (CityRoad)-[:DataRequirement]->(AEB)
MERGE (Pedestrian)-[:DataRequirement]->(AEB)
MERGE (TrafficSign)-[:DataRequirement]->(AEB)

MERGE (Sunny)-[:DataRequirement]->(ACC)
MERGE (Night)-[:DataRequirement]->(ACC)
MERGE (Dawn)-[:DataRequirement]->(ACC)
MERGE (Vehiclesc)-[:DataRequirement]->(ACC)
MERGE (TrafficSign)-[:DataRequirement]->(ACC)
MERGE (Highway)-[:DataRequirement]->(ACC)


MERGE (Sunny)-[:DataRequirement]->(LaneKeeping)
MERGE (Night)-[:DataRequirement]->(LaneKeeping)
MERGE (Dawn)-[:DataRequirement]->(LaneKeeping)
MERGE (Vehicles)-[:DataRequirement]->(LaneKeeping)
MERGE (LaneMark)-[:DataRequirement]->(LaneKeeping)
MERGE (Highway)-[:DataRequirement]->(LaneKeeping)

MERGE (Sunny)-[:DataRequirement]->(RemoteParking)
MERGE (Night)-[:DataRequirement]->(RemoteParking)
MERGE (Dawn)-[:DataRequirement]->(RemoteParking)
MERGE (Vehicles)-[:DataRequirement]->(RemoteParking)
MERGE (LaneMark)-[:DataRequirement]->(RemoteParking)
MERGE (Pedestrian)-[:DataRequirement]->(RemoteParking)



// ML Alogrhitm

MERGE (YOLO)-[:Serve]->(ObjectDetection)
MERGE (PointNet)-[:Serve]->(SemanticSegmantiation)


MERGE (ENet)-[:Serve]->(LaneDetection)
MERGE (TransTrack)-[:Serve]->(ObjectTracking)


MERGE (Social)-[:Serve]->(TrajectoryPrediction)

MERGE (YOLO)-[:Serve]->(SpaceDetection)
MERGE (UNet)-[:Serve]->(SpaceDetection)

// ML Dataflow

MERGE (camera)-[:Collect_Data]->(YOLO)
MERGE (Lidar)-[:Collect_Data]->(PointNet)


MERGE (camera)-[:Collect_Data]->(ENet)
MERGE (camera)-[:Collect_Data]->(UNet)
MERGE (camera)-[:Collect_Data]->(TransTrack)

MERGE (camera)-[:Collect_Data]->(YOLO)

MERGE (camera)-[:Collect_Data]->(Social)


// Return the flow
RETURN *



// Discard Code 

Merge (object:functional {
  name: "Object Detector",
  type: "Functional Blocks"
})

Merge (radar_tracker:functional {
  name: "Radar Tracker",
  type: "Functional Blocks"
})

Merge (lidar_tracker:functional {
  name: "Lidar Tracker",
  type: "Functional Blocks"
})
MERGE (YOLO)-[:Consist]->(AEB)

MERGE (camera)-[:Dataflow]->(AEB)
MERGE (radar)-[:Dataflow]->(AEB)