# Quadcopter Autonomy Curriculum

## Table of Contents

- [[#Overview|Overview]]
- [[#Scope|Scope]]
- [[#Phase 4: Simulation|Phase 4: Simulation]]
- [[#Phase 5: Manual-Flight Hardware Build|Phase 5: Manual-Flight Hardware Build]]
- [[#Phase 6: Autonomous Flight|Phase 6: Autonomous Flight]]
- [[#References|References]]

## Overview

🚁 Stages 4–6 of a staged progression toward autonomous quadcopter flight, following [[curricula/robotics/curriculum|SO-101]] (Stage 1), [[curricula/robotics/mobile-robotics|mobile robotics]] (Stage 2), and [[curricula/robotics/balancing-robots|balancing robots]] (Stage 3). Combined into one document rather than three, since — unlike the earlier stages, which are genuinely different platforms — these three phases are a single continuous build-up on one platform (quadcopters), the same way [[curricula/robotics/curriculum|the SO-101 curriculum]] contains multiple modules in one doc.

This document is a **scaffold, not a fully-researched curriculum** — same caveat as the other new stage docs.

## Scope

The three phases, in order, and why each precedes the next:

1. **Simulation first** — flight-controller software (attitude/rate PID cascades, mixer configuration) tuned and debugged in a simulator, at zero hardware cost and zero crash risk. Standard practice in the field.
2. **Manual-flight hardware build** — a real, cheap micro/FPV-class quadcopter, hand-assembled (frame, ESCs, motors, props, flight-controller board), flown manually. The goal is hardware fluency (soldering, ESC calibration, firmware configuration, crash repair) and stick-flying skill before adding autonomy on top.
3. **Autonomous flight** — a companion computer added to the same or an upgraded airframe, running an autonomy stack (PX4 or ArduPilot + MAVSDK/ROS2), with GPS or visual-inertial odometry for state estimation, executing autonomous missions. This is the capstone of the entire staged progression, and the natural place to bring back the vision/ML skills from [[curricula/robotics/curriculum|Stage 1]] — e.g. vision-based landing or obstacle avoidance.

## Phase 4: Simulation

🔲 Not yet researched. Likely anchor: PX4 SITL (software-in-the-loop) + Gazebo, or a Betaflight/iNav simulator for a more manual-flight-focused starting point before jumping to full PX4/ArduPilot autonomy stacks.

## Phase 5: Manual-Flight Hardware Build

🔲 Not yet researched. Needs: BOM for a cheap FPV/micro-class quad, flight-controller firmware choice (Betaflight is the standard for manual/acro flying; PX4/ArduPilot are the standard for autonomy — worth deciding whether Phase 5's airframe is meant to carry over into Phase 6 or is disposable/practice-only).

## Phase 6: Autonomous Flight

🔲 Not yet researched. Needs: companion computer choice (Raspberry Pi vs. Jetson, weight/power tradeoffs), autonomy stack (PX4 vs. ArduPilot, MAVSDK vs. ROS2 integration), state estimation approach (GPS-only vs. visual-inertial odometry for GPS-denied flight), and a capstone task definition analogous to [[curricula/robotics/curriculum|Stage 1's imitation-learning capstone]].

## References

*(To be populated during the research pass for each phase.)*
