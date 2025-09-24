% =============================================================================
% COMPONENTS & COMPOSITION  (names aligned to Neo4j_output.txt)
% =============================================================================
component('PAEB').
component('Adaptive Cruise Control').
component('Lane Keeping').

% --- Sensors (Neo4j mentions "Mono Camera"; we bind it where needed)
contains('PAEB',                   sensor,  'Mono Camera').
contains('PAEB',                   sensor,  'Lidar').
contains('Adaptive Cruise Control',sensor,  'Mono Camera').
contains('Lane Keeping',           sensor,  'Mono Camera').

% --- Algorithms (from Neo4j: “Include” relations)
contains('PAEB',                   algorithm, 'Object Detection').
contains('PAEB',                   algorithm, 'Semantic Segmantiation').
contains('Adaptive Cruise Control',algorithm, 'Object Tracking').
contains('Adaptive Cruise Control',algorithm, 'Lane Detection').
contains('Lane Keeping',           algorithm, 'Lane Detection').
contains('Lane Keeping',           algorithm, 'Trajectory Prediction').

% --- Actuators (kept for completeness; not in Neo4j_output.txt)
contains('PAEB',                   actuator, brakes).
contains('Adaptive Cruise Control',actuator, brakes).
contains('Adaptive Cruise Control',actuator, throttle).
contains('Lane Keeping',           actuator, brakes).
contains('Lane Keeping',           actuator, throttle).

% =============================================================================
% ALGORITHMS → CAPABILITIES & CONCRETE MODELS  (from Neo4j “Serve” tuples)
% =============================================================================
% What each algorithm can identify (domain semantics kept)
algorithm_identifies('Object Detection',   pedestrian).
algorithm_identifies('Semantic Segmantiation',   environment).
algorithm_identifies('Object Tracking',    vehicle).
algorithm_identifies('Lane Detection',     lane_boundary).
algorithm_identifies('Trajectory Prediction',     road).
% 'Trajectory Prediction' not mapped to a target class in Neo4j_output.txt.

% Concrete implementations named in Neo4j_output.txt
algorithm_uses_model('Object Detection',   'YOLOv5').
algorithm_uses_model('Object Tracking',    'TransTrack').
algorithm_uses_model('Lane Detection',     'ENet-SAD').
algorithm_uses_model('Trajectory Prediction','Social-LSTM').
algorithm_uses_model('Semantic Segmantiation','PointNet++').

% Optionally: sensor → model feeders (from “Collect_Data” tuples)
sensor_feeds_model('Mono Camera', 'YOLOv5').
sensor_feeds_model('Mono Camera', 'ENet-SAD').
sensor_feeds_model('Mono Camera', 'TransTrack').
sensor_feeds_model('Mono Camera', 'Social-LSTM').
sensor_feeds_model('Lidar', 'PointNet++').

% =============================================================================
% REQUIREMENTS (IDs renamed to the SYS-ML-REQ-* labels from Neo4j)
% =============================================================================
% Identities
requirement('SYS-ML-REQ-1').
requirement('SYS-ML-REQ-2').
requirement('SYS-ML-REQ-3').
requirement('SYS-ML-REQ-4').
% (Neo4j lists -4/-5/-6 as well; add if you later model them here.)

% Targets
requirement_target('SYS-ML-REQ-1', pedestrian).
requirement_target('SYS-ML-REQ-2', vehicle).
requirement_target('SYS-ML-REQ-3', lane_boundary).
requirement_target('SYS-ML-REQ-4', road).


% Sensors (Neo4j only lists "Mono Camera")
requirement_sensor('SYS-ML-REQ-1', 'Mono Camera').
requirement_sensor('SYS-ML-REQ-1', 'Lidar').
requirement_sensor('SYS-ML-REQ-2', 'Mono Camera').
requirement_sensor('SYS-ML-REQ-3', 'Mono Camera').
requirement_sensor('SYS-ML-REQ-4', 'Mono Camera').

% Algorithms (aligned names)
requirement_algorithm('SYS-ML-REQ-1', 'Object Detection').
requirement_algorithm('SYS-ML-REQ-1', 'Semantic Segmantiation').
requirement_algorithm('SYS-ML-REQ-2', 'Object Tracking').
requirement_algorithm('SYS-ML-REQ-3', 'Lane Detection').
requirement_algorithm('SYS-ML-REQ-4', 'Trajectory Prediction').

% Models ------------- KEEP THESE TOGETHER -------------
requirement_model('SYS-ML-REQ-1', 'YOLOv5').
requirement_model('SYS-ML-REQ-1', 'PointNet++').

requirement_model('SYS-ML-REQ-2', 'TransTrack').
% requirement_model('SYS-ML-REQ-2', 'PointNet++').

requirement_model('SYS-ML-REQ-3', 'ENet-SAD').
requirement_model('SYS-ML-REQ-4', 'Social-LSTM').

% =============================================================================
% TRACEABILITY LOGIC (unchanged except for renamed symbols)
% =============================================================================
% Component has at least one sensor allowed by the requirement
component_meets_sensor_side(C, Req) :-
    requirement_sensor(Req, S),
    contains(C, sensor, S).

% Component implements ALL algorithms demanded by the requirement
component_meets_algorithm_side(C, Req) :-
    \+ ( requirement_algorithm(Req, A),
         \+ contains(C, algorithm, A)
       ).

% A model is present in a component if some contained algorithm uses it
component_has_model(C, M) :-
    contains(C, algorithm, A),
    algorithm_uses_model(A, M).

% Component implements ALL models demanded by the requirement
component_meets_model_side(C, Req) :-
    \+ ( requirement_model(Req, M),
         \+ component_has_model(C, M)
       ).

% Component (via some algorithm) can identify the target class
component_meets_target_side(C, Req) :-
    requirement_target(Req, Class),
    contains(C, algorithm, A),
    algorithm_identifies(A, Class).

% A component traces to a requirement if:
%  (i) it has ≥1 required sensor,
%  (ii) it implements all required algorithms,
%  (iii) it implements all required models,
%  (iv) some contained algorithm can identify the required target.
traces_to_requirement(C, Req) :-
    component(C), requirement(Req),
    component_meets_sensor_side(C, Req),
    component_meets_algorithm_side(C, Req),
    component_meets_model_side(C, Req),
    component_meets_target_side(C, Req).

% Requirement ↔ sensors explicitly named
req_related_sensor(Req, S) :-
    requirement(Req), requirement_sensor(Req, S).

% Requirement ↔ algorithms both named and capable of the target
req_related_algorithm(Req, A) :-
    requirement(Req),
    requirement_target(Req, Class),
    requirement_algorithm(Req, A),
    algorithm_identifies(A, Class).

% Requirement ↔ models explicitly named
req_related_model(Req, M) :-
    requirement(Req), requirement_model(Req, M).

% Per-component traces of the exact elements that satisfy a requirement
comp_req_sensor(C, Req, S) :-
    traces_to_requirement(C, Req),
    requirement_sensor(Req, S),
    contains(C, sensor, S).

comp_req_algorithm(C, Req, A) :-
    traces_to_requirement(C, Req),
    requirement_algorithm(Req, A),
    contains(C, algorithm, A).

comp_req_model(C, Req, M) :-
    traces_to_requirement(C, Req),
    requirement_model(Req, M),
    component_has_model(C, M).