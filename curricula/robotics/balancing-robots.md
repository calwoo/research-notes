# Balancing Robots Curriculum

## Table of Contents

- [[#Overview|Overview]]
- [[#Scope|Scope]]
- [[#Candidate Hardware|Candidate Hardware]]
- [[#Planned Modules|Planned Modules]]
- [[#References|References]]

## Overview

⚖️ Stage 3 of a staged progression toward autonomous quadcopter flight, following [[curricula/robotics/curriculum|SO-101]] (Stage 1) and [[curricula/robotics/mobile-robotics|mobile robotics]] (Stage 2). This document is a **scaffold, not a fully-researched curriculum** — same caveat as the mobile-robotics stub.

> [!NOTE] Why this stage exists
> A self-balancing two-wheeled robot is the closest ground-based rehearsal for a flight controller that exists. Both are underactuated, inherently unstable systems requiring continuous high-bandwidth feedback control from fused IMU data — a quadcopter's attitude controller and a balancing robot's stabilization loop are structurally the same problem (cascaded PID or LQR on a linearized inverted-pendulum model), except a balancing robot fails by tipping over, not crashing out of the sky.

## Scope

Core topics this stage needs to cover, at minimum:
- IMU fundamentals — accelerometer/gyroscope characteristics, noise, drift
- Sensor fusion for orientation estimation — complementary filter, then Kalman/EKF
- Inverted-pendulum dynamics and linearization
- Cascaded control loops (inner attitude-rate loop, outer angle/position loop) — direct rehearsal for a flight controller's rate/attitude cascade
- Real-time embedded control loop implementation (this stage likely moves off general-purpose Python timing and into a microcontroller with a hard real-time loop — relevant preparation for flight-controller firmware in Stage 4+)

## Candidate Hardware

Not yet decided. Likely candidates to evaluate when this stage starts: a DIY two-wheel balancing robot build (widely documented hobbyist category — search terms like "self-balancing robot Arduino/ESP32 MPU6050"), or a semi-kit option if one exists with good documentation. Given the direct relevance to flight controllers, worth checking whether any balancing-robot kit explicitly markets itself as flight-controller-adjacent (e.g. built around the same class of flight-controller firmware/IMU boards used in Stage 4+, for maximum transfer).

## Planned Modules

🔲 Not yet broken into modules with verified materials.

## References

*(To be populated during the research pass for this stage.)*
