
// Define the system requirement .....
Merge (sys_safe_req1:System_Safety_Requirement {
  name: "SYS-SAF-REQ-1",
  type: "System Safety Requirement"
})

Merge (sys_ml_req1:ML_Safety_Requirement {
  name: "SYS-ML-REQ-1",
  type: "ML Safety Requirement"
})

Merge (sys_ml_req2:ML_Safety_Requirement {
  name: "SYS-ML-REQ-2",
  type: "ML Safety Requirement"
})
Merge (sys_ml_req3:ML_Safety_Requirement {
  name: "SYS-ML-REQ-3",
  type: "ML Safety Requirement"
})
Merge (sys_ml_req4:ML_Safety_Requirement {
  name: "SYS-ML-REQ-4",
  type: "ML Safety Requirement"
})
Merge (sys_ml_req5:ML_Safety_Requirement {
  name: "SYS-ML-REQ-5",
  type: "ML Safety Requirement"
})
Merge (sys_ml_req6:ML_Safety_Requirement {
  name: "SYS-ML-REQ-6",
  type: "ML Safety Requirement"
})


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

Merge (AEB:functionalility {
  name: "PAEB",
  type: "Functional View"
})

Merge (LaneKeeping:functionalility {
  name: "Lane Keeping",
  type: "Functional View"
})

Merge (ACC:functionalility {
  name: "Adaptive Cruise Control",
  type: "Functional View"
})


// Define main ML blocks descriptions.....

Merge (LaneDetection:functionalility {
  name: "Lane Detection",
  type: "Functional View"
})
Merge (ObjectDetection:functionalility {
  name: "Object Detection",
  type: "Functional View"
})
Merge (ObjectTracking:functionalility {
  name: "Object Tracking",
  type: "Functional View"
})
Merge (SemanticSegmantiation:functionalility {
  name: "Semantic Segmantiation",
  type: "Functional View"
})
Merge (TrajectoryPrediction:functionalility {
  name: "Trajectory Prediction",
  type: "Functional View"
})

// Define the ML description.....

    // Object Detection
Merge (YOLO:algorithms {
  name: "YOLOv5",
  type: "ML Component"
})
    // Object Tracking
Merge (TransTrack:algorithms {
  name: "TransTrack",
  type: "ML Component"
})

    // Trajectory Prediction 
Merge (Social:algorithms {
  name: "Social-LSTM",
  type: "ML Component"
})

    // Semantic Segmantiation 
Merge (PointNet:algorithms {
  name: "PointNet++",
  type: "ML Component"
})

    // Lane Tracking
Merge (ENet:algorithms {
  name: "ENet-SAD",
  type: "ML Component"
})



// Create nodes
MERGE (n1:ML_Flow {name: "ML Safety Assurance Scoping", type: "ML Development Flow"})
MERGE (n2:ML_Flow {name: "ML Safety Requirements Assurance", type: "ML Development Flow"})
MERGE (n3:ML_Flow {name: "Data Management Assurance", type: "ML Development Flow"})
MERGE (n4:ML_Flow {name: "Model Learning Assurance", type: "ML Development Flow"})
MERGE (n5:ML_Flow {name: "Model Verification Assurance", type: "ML Development Flow"})
MERGE (n6:ML_Flow {name: "Model Deployment Assurance", type: "ML Development Flow"})

// Create relationships in flow order
MERGE (n1)-[:NEXT]->(n2)
MERGE (n2)-[:NEXT]->(n3)
MERGE (n3)-[:NEXT]->(n4)
MERGE (n4)-[:NEXT]->(n5)
MERGE (n5)-[:NEXT]->(n6)

MERGE (sys_des)-[:Input]->(n1)
MERGE (sys_des)-[:Input]->(n6)
MERGE (sys_safe_req1)-[:Input]->(n1)
MERGE (sys_safe_req1)-[:Input]->(n6)


MERGE (sys_ml_req1)-[:Input]->(n5)
MERGE (sys_ml_req1)-[:Input]->(n3)
MERGE (sys_ml_req1)-[:Input]->(n4)

MERGE (sys_ml_req2)-[:Input]->(n5)
MERGE (sys_ml_req2)-[:Input]->(n3)
MERGE (sys_ml_req2)-[:Input]->(n4)

MERGE (sys_ml_req3)-[:Input]->(n5)
MERGE (sys_ml_req3)-[:Input]->(n3)
MERGE (sys_ml_req3)-[:Input]->(n4)

MERGE (sys_ml_req4)-[:Input]->(n5)
MERGE (sys_ml_req4)-[:Input]->(n3)
MERGE (sys_ml_req4)-[:Input]->(n4)

MERGE (sys_ml_req5)-[:Input]->(n5)
MERGE (sys_ml_req5)-[:Input]->(n3)
MERGE (sys_ml_req5)-[:Input]->(n4)

MERGE (sys_ml_req6)-[:Input]->(n5)
MERGE (sys_ml_req6)-[:Input]->(n3)
MERGE (sys_ml_req6)-[:Input]->(n4)

MERGE (sys_ml_req1)<-[:Output]-(n1)
MERGE (sys_ml_req2)<-[:Output]-(n2)
MERGE (sys_ml_req3)<-[:Output]-(n2)
MERGE (sys_ml_req4)<-[:Output]-(n2)
MERGE (sys_ml_req5)<-[:Output]-(n2)
MERGE (sys_ml_req6)<-[:Output]-(n2)


// System Specifications
MERGE (camera)-[:Consist]->(sys_des)
MERGE (radar)-[:Consist]->(sys_des)
MERGE (Lidar)-[:Consist]->(sys_des)
MERGE (brakes)-[:Consist]->(sys_des)
MERGE (AEB)-[:Consist]->(sys_des)
MERGE (LaneKeeping)-[:Consist]->(sys_des)
MERGE (ACC)-[:Consist]->(sys_des)



// ML Specification
MERGE (ObjectDetection)-[:Include]->(AEB)
MERGE (SemanticSegmantiation)-[:Include]->(AEB)

MERGE (LaneDetection)-[:Include]->(ACC)
MERGE (ObjectTracking)-[:Include]->(ACC)


MERGE (TrajectoryPrediction)-[:Include]->(LaneKeeping)
MERGE (LaneDetection)-[:Include]->(LaneKeeping)

// ML Alogrhitm

MERGE (YOLO)-[:Serve]->(ObjectDetection)
MERGE (PointNet)-[:Serve]->(SemanticSegmantiation)


MERGE (ENet)-[:Serve]->(LaneDetection)
MERGE (TransTrack)-[:Serve]->(ObjectTracking)


MERGE (Social)-[:Serve]->(TrajectoryPrediction)


// ML Dataflow

MERGE (camera)-[:Collect_Data]->(YOLO)
MERGE (Lidar)-[:Collect_Data]->(PointNet)


MERGE (camera)-[:Collect_Data]->(ENet)
MERGE (camera)-[:Collect_Data]->(TransTrack)


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