# Mobile Robotics Curriculum

## Table of Contents

- [[#Overview|Overview]]
- [[#Scope|Scope]]
- [[#Candidate Hardware|Candidate Hardware]]
- [[#Planned Modules|Planned Modules]]
- [[#References|References]]

## Overview

🚗 Stage 2 of a staged progression toward autonomous quadcopter flight, following [[curricula/robotics/curriculum|the SO-101 manipulator curriculum]] (Stage 1). This document is currently a **scaffold, not a fully-researched curriculum** — materials, per-module course substitutes, and hardware pricing have not yet been verified the way Stage 1's were (real course syllabi fetched, vendor packing lists checked, etc.). Fill in with the same rigor when this stage is actually reached.

> [!NOTE] Why this stage exists
> The SO-101 track covers fixed-base manipulation but has no navigation, localization, or mapping content. Autonomous flight is fundamentally a 3D navigation problem, so this stage exists to build the 2D navigation stack (odometry, SLAM, path planning, sensor fusion) somewhere mistakes are cheap — a mobile robot that gets lost just needs to be picked up, unlike a drone.

## Scope

Core topics this stage needs to cover, at minimum:
- Odometry (wheel encoders, dead reckoning, and its drift/error accumulation)
- Localization and mapping (SLAM — at least a working intuition for EKF-SLAM and/or a modern LiDAR/visual SLAM pipeline)
- Path planning (A*, RRT/RRT*, or similar) and obstacle avoidance
- Sensor fusion (combining odometry, IMU, and exteroceptive sensors — LiDAR or depth camera)
- A middleware layer (most likely ROS2, given it's the de facto standard and will also be relevant for the quadcopter-autonomy stage)

## Candidate Hardware

Not yet decided — two directions worth weighing when this stage starts:
- **[LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi)** — LeRobot-native mobile base (essentially an SO-101-family arm on a mobile platform), ~$300–500 depending on configuration per the [[curricula/robotics/so101-arm-buildout|earlier vendor research]]'s companion ecosystem. Advantage: stays inside the LeRobot/Feetech-servo ecosystem already invested in, and reuses SO-101 skills directly. Disadvantage: LeRobot's own tooling is manipulation/imitation-learning focused, not classical SLAM/path-planning focused, so the "mobile robotics" content proper (localization, mapping) may need to be layered on top rather than coming for free.
- **A generic ROS2 differential-drive rover** (e.g. TurtleBot-class or a DIY chassis + LiDAR + Raspberry Pi/Jetson) — advantage: lands directly in the mainstream ROS2/Nav2 ecosystem with the most mature tutorials, documentation, and community support for exactly this stage's scope. Disadvantage: a second, unrelated hardware/software ecosystem alongside LeRobot.

## Planned Modules

🔲 Not yet broken into modules with verified materials. Expect this to mirror Stage 1's structure once researched: concept goals → materials (course/textbook substitutes, verified against real syllabi) → milestones tied to the physical hardware.

## References

*(To be populated during the research pass for this stage.)*
