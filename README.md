# ROS2 Gate Vision Pipeline

This is the ROS2 implementation of the gate vision pipeline. It is intended to be used in a ROS2 environment and should be integrated into a larger ROS2 system.

## Dependencies

- vision-pipeline-interfaces: Custom ROS2 interfaces for the gate vision pipeline.
- gate-vision-model: The gate vision model which contains:
  - YOLO model for gate detection.
  - MobileNetV3 model for corner detection.
  - Combined inference script for gate and corner detection.