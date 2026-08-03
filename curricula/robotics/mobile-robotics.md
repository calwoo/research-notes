# Mobile Robotics Curriculum

## Table of Contents

- [[#Overview|Overview]]
- [[#Pre-Stage-2 Warm-up: XRP|Pre-Stage-2 Warm-up: XRP]]
- [[#Scope|Scope]]
- [[#Candidate Hardware|Candidate Hardware]]
- [[#Planned Modules|Planned Modules]]
- [[#References|References]]

## Overview

🚗 Stage 2 of a staged progression toward autonomous quadcopter flight, following [[curricula/robotics/curriculum|the SO-101 manipulator curriculum]] (Stage 1). Hardware is now decided and the module structure is drafted (see [[#Candidate Hardware|Candidate Hardware]] and [[#Planned Modules|Planned Modules]] below, and the companion [[curricula/robotics/mobile-robotics-buildout|buildout doc]]) — remaining work is verifying exact BOM pricing/stock immediately before ordering and filling in per-module reading materials with the same rigor as Stage 1's CMU-course-substitute research.

> [!NOTE] Why this stage exists
> The SO-101 track covers fixed-base manipulation but has no navigation, localization, or mapping content. Autonomous flight is fundamentally a 3D navigation problem, so this stage exists to build the 2D navigation stack (odometry, SLAM, path planning, sensor fusion) somewhere mistakes are cheap — a mobile robot that gets lost just needs to be picked up, unlike a drone.

## Pre-Stage-2 Warm-up: XRP

🛞 Before committing to Stage 2's own hardware decision (below), warm up on odometry, encoders, and proportional control — the cheapest, lowest-risk fundamentals in this stage's [[#Scope|Scope]] — using WPI's **XRP** (Experiential Robotics Platform).

**Hardware:** [XRP Kit](https://www.sparkfun.com/experiential-robotics-platform-xrp-kit.html) — **$119.95** from SparkFun. Controller is a Raspberry Pi **RP2350B** (dual Cortex-M33, 16MB flash, 8MB PSRAM) with an onboard 6-DoF IMU (LSM6DSOX), two Qwiic connectors, and dual-channel motor drivers for up to four motors. The kit itself ships with 2 encoded drive motors + 2 casters (differential drive, not a balancing platform — relevant only to this stage, not [[curricula/robotics/balancing-robots|Stage 3]]), an ultrasonic rangefinder, a line-following sensor, a servo, and a 3D-printed chassis. Batteries and USB-C cable are not included. Programmed with **WPILib** — the same toolchain used for full-size FIRST Robotics Competition (FRC) robots — via Blockly or Python.

**Learning materials — two curricula exist, pick based on self-study fit:**
- [Introduction to Robotics](https://introtoroboticsv2.readthedocs.io/en/latest/course/course_info/index.html) (WPI Global STEM Education Initiative, ReadTheDocs) — **preferred primary path**, since it's structured for individual self-paced learners rather than a classroom. Modules run Introduction → Driving → Sensors → Manipulator (a small robot arm attachment — direct conceptual overlap with SO-101 skills, just at toy scale) → Capstone (an "autonomous delivery robot" challenge). Each module is a sequence of ordered, interactive challenges on the physical robot.
- [WPI XRP Curriculum](https://wp.wpi.edu/xrp/curriculum/) (wp.wpi.edu) — the original K-12 classroom curriculum, four units with full lesson-plan PDFs (slides, guided notes, homework, answer keys, grading rubrics): [Unit 1](https://wp.wpi.edu/xrp/curriculum/unit-1/) (assembly, Blockly/Python basics, Gate Maze Challenge), [Unit 2](https://wp.wpi.edu/xrp/curriculum/unit-2/) (ultrasonic sensing, proportional control, encoders, Moving Maze), [Unit 3](https://wp.wpi.edu/xrp/curriculum/unit-3/) (line following — reflectance + ultrasonic, on/off vs. proportional control), [Unit 4](https://wp.wpi.edu/xrp/curriculum/unit-4/) (servo integration, final time-trial capstone). Useful as a reference for classroom-tested exercise design even if not followed lesson-by-lesson.
- [WPILib XRP docs](https://docs.wpilib.org/en/stable/docs/xrp-robot/index.html) — programming reference: [hardware & imaging](https://docs.wpilib.org/en/stable/docs/xrp-robot/hardware-and-imaging.html), [getting to know your XRP](https://docs.wpilib.org/en/stable/docs/xrp-robot/getting-to-know-xrp.html), [hardware support](https://docs.wpilib.org/en/stable/docs/xrp-robot/hardware-support.html), [web UI](https://docs.wpilib.org/en/stable/docs/xrp-robot/web-ui.html), [programming the XRP](https://docs.wpilib.org/en/stable/docs/xrp-robot/programming-xrp.html).
- [experiential.bot](https://experiential.bot/) — community hub: forum, helpdesk, and 3D-printable hardware extensions ([XBS build system](https://www.printables.com/model/1452225-xbs-xrp-experiential-robotics-build-system-alpha-r), [legacy parts](https://www.printables.com/model/1216372-xrp-robot-kit)).

> [!WARNING] Gap vs. this stage's Scope
> Neither XRP curriculum covers SLAM, path planning, sensor fusion, or ROS2 — WPILib targets FRC-style teleop/autonomous-period robots, not a navigation stack. XRP's role here is strictly the **odometry + encoder + proportional-control warm-up**; the actual SLAM/ROS2/path-planning content comes from [[#Candidate Hardware|the DIY ROS2 rover decided on below]] once the warm-up is done.

> [!QUESTION] Open question: layering ROS2 onto the same XRP hardware instead of switching platforms?
> [micro-ROS](https://github.com/micro-ROS/micro_ros_raspberrypi_pico_sdk) does run on RP2040/RP2350 in principle, which raises the option of keeping XRP as the one physical platform for this whole stage — micro-ROS for the low-level motor/encoder/IMU interface, talking to ROS2 nodes (Nav2, SLAM) on a companion computer. But this means replacing the WPILib firmware entirely (losing the curriculum above) and there's no confirmed existing integration bridging XRP's specific hardware to micro-ROS out of the box — unverified, would need a spike before committing. Cheaper default: treat XRP purely as the warm-up and pick separate ROS2-native hardware for the real Stage 2 build.

## Scope

Core topics this stage needs to cover, at minimum:
- Odometry (wheel encoders, dead reckoning, and its drift/error accumulation)
- Localization and mapping (SLAM — at least a working intuition for EKF-SLAM and/or a modern LiDAR/visual SLAM pipeline)
- Path planning (A*, RRT/RRT*, or similar) and obstacle avoidance
- Sensor fusion (combining odometry, IMU, and exteroceptive sensors — LiDAR or depth camera)
- A middleware layer (most likely ROS2, given it's the de facto standard and will also be relevant for the quadcopter-autonomy stage)

## Candidate Hardware

> [!NOTE] XRP is warm-up, not a Stage 2 candidate
> The [[#Pre-Stage-2 Warm-up: XRP|XRP]] above covers odometry/encoder/proportional-control fundamentals but not SLAM, path planning, or ROS2 — the decision below is the real Stage 2 hardware call and is unaffected by the XRP warm-up.

**Decision made — see [[curricula/robotics/mobile-robotics-buildout|Mobile Robot Buildout doc]] for the full comparison and BOM.** Going with a **DIY ROS2 rover**, following the free [Articulated Robotics tutorial series](https://articulatedrobotics.xyz/tutorials/) (Josh Newans): source parts yourself (Raspberry Pi 4B/5, RPLiDAR A1, Arduino Nano for motor control, brushed DC gearmotors with encoders, camera), assemble, and write the `ros2_control` hardware interface from scratch — same hands-on-first reasoning as the SO-101's Path C decision. Estimated cost **~$275–350**, versus $900+ for either official TurtleBot platform.

Two other directions were weighed and ruled out (detail in the buildout doc):
- **[LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi)** — LeRobot-native mobile base, ~$300–500. Stays inside the LeRobot/Feetech ecosystem already invested in, but its tooling is manipulation-focused, not SLAM/path-planning focused — the actual Stage 2 content would need to be layered on top rather than coming for free.
- **TurtleBot 3 / TurtleBot 4** — official ROS2 reference platforms with excellent documentation, but $900–$1,900 and largely pre-integrated, which removes exactly the hardware-integration learning value this stage is for.

## Planned Modules

Derived directly from the Articulated Robotics tutorial series structure (see [[curricula/robotics/mobile-robotics-buildout#Curriculum Mapping|Curriculum Mapping]] in the buildout doc for the full diagram):

- [ ] **Concept & Design** — URDF modeling of the rover, Gazebo simulation (zero-hardware-cost first pass, same "simulate first" principle as the quadcopter stage's Phase 4)
- [ ] **Hardware Bring-up** — Raspberry Pi OS/ROS2 setup, power system design, LiDAR integration (RPLiDAR A1 → `LaserScan` messages), camera integration
- [ ] **`ros2_control`** — hardware interface concepts, then wiring it to the real robot (encoders → odometry)
- [ ] **Teleoperation** — manual driving via ROS2 topics, sanity-checks the full hardware stack before autonomy
- [ ] **SLAM** — `slam_toolbox` against the RPLiDAR A1, building a map of a real space
- [ ] **Nav2** — path planning and autonomous navigation using the map from the SLAM module
- [ ] **Object Tracking** — OpenCV-based detection/following using the camera, a stretch module tying back into Stage 1's computer-vision skills

**Milestone:** the rover builds and holds a live SLAM map of a real room, then autonomously navigates to a commanded goal pose via Nav2 — directly analogous to the SO-101's teleoperation-to-autonomous-policy arc, but for navigation instead of manipulation.

> [!WARNING] Sensor fusion gap
> No IMU-fusion step was found in the tutorial series' episode list (see the buildout doc's Open Questions). This stage's Scope requires sensor fusion beyond LiDAR+camera — flag for a supplementary source once this module is reached.

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| XRP (Experiential Robotics Platform) Kit — SparkFun | Hardware source and specs for the pre-Stage-2 warm-up robot, $119.95, RP2350B controller | [sparkfun.com](https://www.sparkfun.com/experiential-robotics-platform-xrp-kit.html) |
| Introduction to Robotics (WPI Global STEM, ReadTheDocs) | Self-study-friendly XRP curriculum: driving, sensors, manipulator, capstone delivery-robot challenge | [introtoroboticsv2.readthedocs.io](https://introtoroboticsv2.readthedocs.io/en/latest/course/course_info/index.html) |
| XRP Curriculum (WPI) | Original K-12 classroom curriculum, 4 units with full lesson-plan PDFs; used as exercise-design reference | [wp.wpi.edu/xrp/curriculum](https://wp.wpi.edu/xrp/curriculum/) |
| WPILib XRP Documentation | Programming reference (hardware, sensors, Web UI, Python/Blockly) — same toolchain as full FRC robots | [docs.wpilib.org](https://docs.wpilib.org/en/stable/docs/xrp-robot/index.html) |
| experiential.bot | XRP community hub — forum, helpdesk, 3D-printable hardware extensions | [experiential.bot](https://experiential.bot/) |
| micro-ROS on Raspberry Pi Pico SDK | Investigated for the open question of running ROS2 directly on XRP's RP2350B instead of switching hardware for Stage 2 proper | [github.com/micro-ROS](https://github.com/micro-ROS/micro_ros_raspberrypi_pico_sdk) |
| Mobile Robot Buildout doc | Full hardware decision, BOM, and curriculum mapping for the Stage 2 DIY ROS2 rover | [[curricula/robotics/mobile-robotics-buildout\|Mobile Robot Buildout]] |
| Articulated Robotics — "Build a Mobile Robot with ROS" (Josh Newans) | Primary tutorial series anchoring the DIY rover build and this stage's module structure | [articulatedrobotics.xyz](https://articulatedrobotics.xyz/tutorials/) |
