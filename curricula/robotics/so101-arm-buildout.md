# SO-101 Robot Arm: Build vs. Buy, and a Robotics Learning Path

## Sources

| Source | Type | Key Contribution | Link |
|--------|------|-------------------|------|
| LeRobot SO-101 docs (Hugging Face) | doc | Official assembly steps, motor gear-ratio table, calibration/teleop workflow | [huggingface.co/docs/lerobot/en/so101](https://huggingface.co/docs/lerobot/en/so101) |
| SO-ARM100/SO-101 repo (TheRobotStudio) | repo / BOM | Canonical bill of materials with per-region vendor links, 3D printing specs (material, nozzle, infill) | [github.com/TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) |
| SO-101 Specs & Setup Guide (SVRC) | blog | General pricing/spec overview, cross-check against official BOM | [roboticscenter.ai/hardware/so-101](https://www.roboticscenter.ai/hardware/so-101) |
| TechEBlog / Hackster.io coverage of SO-101 launch | blog | Independent cost estimate (~$100-130 non-printed parts) at launch | [techeblog.com](https://www.techeblog.com/hugging-face-so-101-3d-printable-robotic-arm/), [hackster.io](https://www.hackster.io/news/hugging-face-launches-the-so-101-an-upgraded-low-cost-3d-printable-autonomous-robot-arm-532360f441eb) |
| Seeed Studio SO-ARM101 Pro Motor Kit | vendor listing | Motors-only kit pricing (US warehouse $288.99); separate $35 3D-printed-parts add-on | [seeedstudio.com](https://www.seeedstudio.com/SO-ARM101-Low-Cost-AI-Arm-Kit-Pro-p-6427.html) |
| CNX Software: SO-ARM101 kit overview | blog | Resolved Seeed's Standard ($220) vs Pro ($240) pair pricing and servo-torque difference, independent of US-warehouse markup | [cnx-software.com](https://www.cnx-software.com/2025/05/02/so-arm101-open-source-dual-robotic-arm-kit-works-with-hugging-faces-lerobot/) |
| WowRobo SO-ARM101 DIY Kit & Assembled | vendor listing | Tiered packages: printed-parts-only, unassembled pair+camera, assembled pair+camera (all listed $199) | [shop.wowrobo.com](https://shop.wowrobo.com/products/so-arm101-diy-kit-assembled-version-1) |
| PartaBot SO-ARM101 | vendor listing | US-based option, electronics-only vs. full-kit, assembled vs. unassembled toggle | [partabot.com](https://partabot.com/products/so-arm101) |
| Hiwonder LeRobot SO-ARM101 (official manufacturer) | vendor listing | Tiered kits: DIY/Starter/Standard/Advanced, camera included from Standard tier up | [hiwonder.com](https://www.hiwonder.com/products/lerobot-so-101) |
| OZ Robotics — Hiwonder SO-ARM101 (DIY Kit/Unassembled) | vendor listing (US-reachable reseller) | Unassembled pair with dual camera (gripper + external) bundled | [ozrobotics.com](https://ozrobotics.com/shop/hiwonder-lerobot-so-arm101-open-source-6-axis-robotic-arm-diy-kit-unassembled/) |
| ThinkRobotics SO-ARM101 DIY Kit | vendor listing (India-focused) | Cross-check price point (₹27,999.99 ≈ $325); not favored for US shipping | [thinkrobotics.com](https://thinkrobotics.com/products/so-arm101-hugging-face-lerobot) |

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

## Vendor Comparison — Unassembled Kits (Path C)

Decision made: going with **Path C, unassembled**, so the arm is built and wired by hand rather than pre-assembled. Comparison below is scoped to unassembled/DIY-tier listings only — pair pricing, US buyer. Sorted by all-in cost (kit + camera if not bundled; ~$25 for a basic USB webcam where needed).

| Vendor | Product | Kit price | Stock (verified) | Shipping (US) | Camera | All-in est. incl. shipping |
|---|---|---|---|---|:---:|---|
| **Hiwonder (official manufacturer)** | DIY Kit / Unassembled | $269.99 | ✅ in stock | FOB Shenzhen, DHL/UPS/FedEx, ~6 business days; order is under the $499 free-shipping threshold, so shipping is charged — est. ~$70 by comparison with OZ Robotics' confirmed rate | ❌ (+~$25 webcam) | **~$365** |
| **Seeed Studio (Pro)** | Motor Kit + 3D Printed Skeleton add-on | $288.99 + $35 | ✅ in stock (Pro only — the $220/$240 Standard/Pro pricing reported by CNX Software appears to be AliExpress/international, **not** confirmed live on seeedstudio.com, which shows only the Pro tier at $288.99) | "US Warehouse" option shown at checkout; unconfirmed whether this means cheaper/faster domestic shipping or just faster dispatch of the same international price | ❌ (+~$25 webcam) | **~$349** (optimistic — depends on unconfirmed US-warehouse shipping) |
| **OZ Robotics** (Hiwonder reseller) | DIY Kit/Unassembled | $306.99 | Stock status not confirmed on page | Confirmed $72.99 standard shipping, China, 4–7 business days | ✅ (2: gripper + external) | **$379.98** (only fully-confirmed shipping number of the three) |
| WowRobo | Package 2: Unassembled Kit | $259 | ❌ sold out (variant id `46588630630617`, corrects an earlier — wrong — $199 figure from this thread) | Ships China via UPS/DHL; vendor states duties/taxes prepaid for US buyers; cost unconfirmed | ✅ (1) | N/A — not currently orderable |
| PartaBot (US-based) | Electronics-only / Full-kit toggle | List $329, seen on sale at $119 | ❌ sold out at research time | Unknown | Unclear | Unclear |
| ThinkRobotics (India) | DIY Kit | ₹27,999.99 (≈$325) | Unconfirmed | Not competitive for US shipping | ❌ | ~$350+ |

**Read:** once shipping is factored in, Hiwonder and Seeed Pro converge to roughly the same ~$350–365 total, with OZ Robotics clearly the most expensive at ~$380 (its shipping cost is the only one directly confirmed, which also makes it the most trustworthy number in this table). WowRobo remains structurally the cheapest kit but is not currently purchasable.

## Synthesis

Given the stated priority (hands-on learning over pure speed or pure cost) and no printer access: the dominant learning value in this build is in **mechanical assembly, servo bus wiring, motor-ID/EEPROM configuration, and calibration** — not in operating an FDM printer or researching individual BOM vendors. Those are real skills, but tangential to robotics specifically.

**Decision: Path C, unassembled.** Buy a complete unassembled kit (pre-printed parts + motors, ideally camera bundled) and do 100% of the assembly, wiring, motor configuration, and calibration by hand. This captures ~90% of the hands-on value of Path A at lower logistics overhead and comparable-or-lower cost, and defers the 3D-printer purchase decision to later — once it's clear whether more hardware projects are coming, that's a better-informed $200–250 call than making it upfront.

**Final vendor recommendation: Hiwonder DIY Kit ($269.99).** Once shipping is included, the field converges to within ~$30 of each other, so the deciding factors are practical rather than price: Hiwonder is the only option that is simultaneously (a) confirmed in stock right now, (b) sold by the actual manufacturer of the design — best assurance on part fit and servo QC, (c) a single consolidated order, with no separate printed-parts line item or warehouse selector to coordinate, and (d) shipped via trackable major carriers (DHL/UPS/FedEx) in ~6 business days. All-in with an estimated shipping cost and a separately-sourced webcam, it lands around $365 — no worse than the alternatives, and simpler to execute.

**Backup:** Seeed Studio Pro (~$349 all-in, optimistic) if the US-warehouse shipping turns out to be cheap/fast and the extra step of a two-part order (motor kit + $35 print add-on) isn't a hassle. Revisit WowRobo if it restocks — it's structurally the cheapest kit and bundles a camera.

> [!QUESTION] Open question: printer purchase timing
> Path B (buy a printer now) was deprioritized on hands-on-learning grounds, but if the broader robotics track ends up involving custom end-effectors, sensor mounts, or other iterative hardware, owning a printer becomes valuable sooner rather than later. Revisit after the first arm is built.

## Open Questions

- Get an exact Hiwonder shipping quote (checkout is address-dependent; ~$70 used above is an estimate by comparison to OZ Robotics' confirmed rate, not a confirmed Hiwonder number).
- Confirm whether Seeed Studio's "US Warehouse" checkout option changes shipping cost/time or just dispatch location, and whether the $220/$240 Standard/Pro pricing (CNX Software, presumably AliExpress) is actually purchasable anywhere reachable from the US at that price.
- Live stock check needed on WowRobo's Package 2 ($259, variant id `46588630630617`) and PartaBot before ordering — both showed "sold out" at research time.
- Once the arm is built: what's the first task/project to target for imitation learning (informs camera setup, gripper choice, workspace design)?
