# SO-101 Robot Arm: Build vs. Buy, and a Robotics Learning Path

## Sources

| Source | Type | Key Contribution | Link |
|--------|------|-------------------|------|
| LeRobot SO-101 docs (Hugging Face) | doc | Official assembly steps, motor gear-ratio table, calibration/teleop workflow | [huggingface.co/docs/lerobot/en/so101](https://huggingface.co/docs/lerobot/en/so101) |
| SO-ARM100/SO-101 repo (TheRobotStudio) | repo / BOM | Canonical bill of materials with per-region vendor links, 3D printing specs (material, nozzle, infill) | [github.com/TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) |
| SO-101 Specs & Setup Guide (SVRC) | blog | General pricing/spec overview, cross-check against official BOM | [roboticscenter.ai/hardware/so-101](https://www.roboticscenter.ai/hardware/so-101) |
| TechEBlog / Hackster.io coverage of SO-101 launch | blog | Independent cost estimate (~$100-130 non-printed parts) at launch | [techeblog.com](https://www.techeblog.com/hugging-face-so-101-3d-printable-robotic-arm/), [hackster.io](https://www.hackster.io/news/hugging-face-launches-the-so-101-an-upgraded-low-cost-3d-printable-autonomous-robot-arm-532360f441eb) |
| Seeed Studio SO-ARM101 Pro Motor Kit | vendor listing | Motors-only kit pricing ($288.99); separate $35 3D-printed-parts add-on | [seeedstudio.com](https://www.seeedstudio.com/SO-ARM101-Low-Cost-AI-Arm-Kit-Pro-p-6427.html) |
| WowRobo SO-ARM101 DIY Kit & Assembled | vendor listing | Tiered packages: printed-parts-only, unassembled pair+camera, assembled pair+camera (all listed $199) | [shop.wowrobo.com](https://shop.wowrobo.com/products/so-arm101-diy-kit-assembled-version-1) |
| PartaBot SO-ARM101 | vendor listing | US-based option, electronics-only vs. full-kit, assembled vs. unassembled toggle | [partabot.com](https://partabot.com/products/so-arm101) |
| Hiwonder LeRobot SO-ARM101 (official manufacturer) | vendor listing | Tiered kits: DIY/Starter/Standard/Advanced, camera included from Standard tier up | [hiwonder.com](https://www.hiwonder.com/products/lerobot-so-101) |

## Context and Motivation

Starting point for a robotics learning track, grounded in an undergraduate math background. First concrete decision: how to acquire the [SO-101](https://huggingface.co/docs/lerobot/en/so101) arm (Hugging Face's LeRobot reference platform) — source parts and 3D print from scratch, or buy a kit. No 3D printer access currently; priority is hands-on learning value from the build itself, not just fastest time-to-arm.

## Platform Basics

The SO-101 system is a **leader-follower teleoperation pair**:

- **Follower** — the working arm. Most joints use `STS3215` servos with 1/345 gearing (high torque, holds position under load).
- **Leader** — a puppet arm with deliberately weaker/mixed gearing (1/191, 1/345, 1/147 across the six joints — see table below) so it can be backdriven by hand with minimal resistance.

| Leader-Arm Axis | Motor | Gear Ratio |
|---|:---:|:---:|
| Base / Shoulder Pan | 1 | 1/191 |
| Shoulder Lift | 2 | 1/345 |
| Elbow Flex | 3 | 1/191 |
| Wrist Flex | 4 | 1/147 |
| Wrist Roll | 5 | 1/147 |
| Gripper | 6 | 1/147 |

You physically move the leader; its joint encoder positions stream over USB (Feetech half-duplex TTL bus) as target commands to the follower. LeRobot's imitation-learning workflow records `(camera frames + joint state, action)` pairs from this teleoperation, then trains a policy (ACT, diffusion policy, or a VLA) to reproduce the demonstrations.

> [!NOTE] Why not just buy a follower alone
> A follower-only setup blocks the core data-collection loop — you'd be restricted to running datasets other people already recorded, or scripting joint targets directly via the Python API. The marginal cost of the leader arm (~$80–115 in a DIY build) is small relative to what it unlocks, so the working assumption for this thread is **acquire the full pair**.

## Cost Comparison

Three viable paths, given no current 3D printer access:

| Path | What you do yourself | Est. cost (full pair, US) | Learning captured |
|---|---|---|---|
| **A. Full DIY sourcing** — buy motors/electronics per the [official BOM](https://github.com/TheRobotStudio/SO-ARM100), pay a print-on-demand service for structural parts | Source ~8 BOM line items across vendors, pay for printing, assemble, wire, calibrate | ~$230 (electronics, official BOM total for 2 arms) + ~$70 (printed parts ×2) + ~$25 (camera) ≈ **$325** | Everything, plus vendor/procurement logistics |
| **B. Buy a printer + DIY BOM** | Same as A, but print parts yourself | ~$230 (electronics) + ~$200–250 (entry-level printer) ≈ **$430–480** | Everything + FDM printing skill; printer reusable for future projects |
| **C. Complete unassembled kit** (pre-printed parts + motors + camera, single vendor) | Assemble, wire, configure motor IDs (`lerobot-setup-motors`), calibrate | **~$200–300**, one shipment | Same assembly/wiring/calibration as A, minus the printing step and multi-vendor sourcing |

**Official BOM total (2 arms, US pricing, electronics only — no printed parts, no camera):** $229.88, per [TheRobotStudio's BOM](https://github.com/TheRobotStudio/SO-ARM100) (12× STS3215 servos, 2× motor control boards, USB-C cables, power supplies, table clamps, screwdriver set). Single-follower-only equivalent: ~$122.

**3D printing spec (if self-printing):** PLA+, 0.4mm nozzle / 0.2mm layer height (or 0.6mm nozzle / 0.4mm layer), 15% infill. Verified on Prusa MINI+, Creality Ender 3, Bambu Lab systems. Minimum bed size ~220×220mm.

**Vendor kit pricing found so far** (volatile — several listings showed "sold out" at time of research, verify before ordering):

| Vendor | Package | Price | Includes |
|---|---|---|---|
| WowRobo | Package 2: Unassembled Kit | $199 | 1 leader + 1 follower + 1 camera, printed parts + 12 servos |
| WowRobo | Package 3: Assembled | $199 | Same as above, pre-assembled |
| Hiwonder | DIY Kit / Unassembled | $269.99 | Pair, **no camera** |
| Hiwonder | Standard/Advanced Assembled | $269.99+ | Pair + camera(s), pre-assembled |
| Seeed Studio | Pro Motor Kit | $288.99 | Motors only — **unclear if per-arm or per-pair**, flagged as open question |
| Seeed Studio | 3D Printed Parts add-on | $35 | Printed structural parts (quantity/arm-count unclear) |
| PartaBot | Electronics-only / Full-kit toggle | List price $329, sale $119 (sold out at research time) | Pair; exact tier pricing unclear |

## Synthesis

Given the stated priority (hands-on learning over pure speed or pure cost) and no printer access: the dominant learning value in this build is in **mechanical assembly, servo bus wiring, motor-ID/EEPROM configuration, and calibration** — not in operating an FDM printer or researching individual BOM vendors. Those are real skills, but tangential to robotics specifically.

**Working recommendation: Path C.** Buy a complete unassembled kit (pre-printed parts + motors + camera, single vendor) and do 100% of the assembly, wiring, motor configuration, and calibration by hand. This captures ~90% of the hands-on value of Path A at lower logistics overhead and comparable-or-lower cost, and defers the 3D-printer purchase decision to later — once it's clear whether more hardware projects are coming, that's a better-informed $200–250 call than making it upfront.

Concrete next step: verify current stock/pricing on Hiwonder's DIY kit and WowRobo's unassembled pair+camera bundle, since availability at these vendors is volatile.

> [!QUESTION] Open question: Seeed Studio kit scope
> Could not confirm from the product page whether the $288.99 "Pro Motor Kit" and $35 "3D Printed Parts" add-on are priced per single arm or for the full leader+follower pair. Matters for the cost comparison — worth checking their wiki or contacting them directly before treating Seeed as a baseline.

> [!QUESTION] Open question: printer purchase timing
> Path B (buy a printer now) was deprioritized on hands-on-learning grounds, but if the broader robotics track ends up involving custom end-effectors, sensor mounts, or other iterative hardware, owning a printer becomes valuable sooner rather than later. Revisit after the first arm is built.

## Open Questions

- Is the Seeed Studio motor-kit pricing per arm or per pair? (see above)
- Current stock status at WowRobo, PartaBot, and Hiwonder — several listings were sold out at research time.
- What's the actual print time / filament cost if going the print-on-demand route (Path A), to sharpen the A vs. C cost comparison?
- Once the arm is built: what's the first task/project to target for imitation learning (informs camera setup, gripper choice, workspace design)?
