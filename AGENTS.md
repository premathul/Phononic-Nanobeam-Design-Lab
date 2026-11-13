# Development instructions

Research owner: **Athul Prem** | Project tag: **_AP_pnc**.

## Current direction

Read README.md, docs/AI_HANDOFF_AP_pnc.md, docs/IMPLEMENTATION_PLAN_AP_pnc.md and docs/PHYSICS_SPECIFICATION_AP_pnc.md. Current user authorization supersedes the historical handoff: architecture can change freely; prioritize the latest reliable physics and a usable configurable simulator. Full archive recovery is not a prerequisite for new development. Preserve actual evidence and label new implementation independently.

The active solver is `simulate_AP_pnc.py` and `src/phononic/`. Root `run_AP_pnc.py` remains a legacy readiness checker. Finite-body modes, Bloch dispersion and power-normalized uniform/perforated scattering are implemented. Two-hole power checks and a bounded 1 GHz complex-response mesh-pair check pass. In the earlier unstructured Gmsh series for the 26-hole device, h200/h160 failed both mesh checks, and h160/h130 passed power but failed complex response. The newer layered series uses a separate port discretization and has its own results below. Reuse saved evidence before further solves or spectral and resonance claims.

The optional English Qt interface is `workbench_AP_pnc.py` (`Launch_Workbench_AP_pnc.cmd` on Windows); its headless worker calls the same solver and mesh-preflight APIs. UI dependencies are separate in `requirements_ui_AP_pnc.txt`. Keep mesh readiness, successful computation and scientific convergence distinct in the interface.

The workbench opens saved runs and compares scattering results through `tools/compare_scattering_AP_pnc.py` without new solves. Identical discretizations are repeatability checks, and failed input power checks cannot establish mesh-pair acceptance. The saved full-device/port follow-up is checked with `tools/check_device_followup_AP_pnc.py`; a successful integrity check does not mean the scientific convergence criteria passed.

The additional `layered` mesh uses `src/phononic/layered_AP_pnc.py` and `configs/perforated_layered_AP_pnc.json`. Its in-plane size and `thickness_layers` are separate controls; z layers also define the port thickness mesh. This remains a full 3D displacement model. `tools/diagnose_wave_resolution_AP_pnc.py` and the saved directional diagnostic establish sensitivity, not complete-device convergence. Keep the existing unstructured Gmsh path and its numerical evidence unchanged.

The tested two-hole 1 GHz thickness/port follow-up is complete. `configs/perforated_layered_reference_AP_pnc.json` selects h80/four layers/100 nm port width; final thickness and width increments and agreement with the saved unstructured reference pass 0.01. `tools/compare_uniform_reference_AP_pnc.py` compares saved uniform runs to the portable finer SAFE reference; coarse native-port phase alone can mislead. These bounded two-hole/600 nm-section checks do not certify the longer 26-hole/1000 nm-section device. Read the saved follow-up before new solves.

The full-device section study in `verification/full_device_port_resolution_AP_pnc.json` supplies a finite 20-by-6 reference with passing independent width and thickness phase increments at 1 GHz over 16.65 µm of uniform guide. The economical 12-by-4 section meets the 0.01 rad check over 3 µm of combined lead length and exceeds it over 16.65 µm. The layered h200, h160 and h130 body solves cover all five left-incident channels at 1 GHz with identical 675-DOF ports, four thickness layers and 32 segments per hole rim. Maximum power changes are 0.00261746 and 0.000848104 for successive pairs, both passing 0.01. Maximum complex changes are 0.21728723 and 0.10704320, both failing. The latest pair's maximum transmission phase shift is 0.107183 rad. The h130 solve has 225747 body DOFs, maximum original relative residual 3.35e-11 and independent flux imbalance 1.15e-12. Public results are under `results/runs/full_device_layered_preflight/`, with suffixes `44ad87c9837d`, `ec5ad09e287e` and `98860bb38cf7`. The exact h130 operator and solution are under `results/runs/full_device_resource_profile/full_h130_operator_84xaqgtd/`. Reuse the results and the checks in `verification/solver_resources_AP_pnc.json`.

The completed 3 µm uniform h160/h130 controls reused those ports. Their maximum native complex transmission error fell from 0.03054 to 0.01497, with a maximum same-channel shift of 0.01836 rad. Channels 2, 4 and 5 showed the largest phase shifts in both uniform and patterned comparisons. The finite reference and end effects limit the diagnostic to the tested guide. Keep patterned phases at their original end planes. The next bounded calculation is one 3 µm uniform h100 control with the same ports and thickness layers, reusing the saved h130 guide for comparison. No h100 run has started. Use native and mesh-pair phase differences to choose the next in-plane target before preparing another full-device mesh.

## Project language

Use English for all project artifacts, including code comments, user-interface text, generated reports and other output text, documentation, and GitHub communications. Standard scientific Unicode symbols and notation are welcome. Chinese may be used in direct conversation and progress updates to the user.

## Work discipline

- Recover actual running state, local changes, current source hashes and saved checks before resuming. Do not restart completed simulations without a reason.
- Deliver a small complete workflow before expanding a campaign. Review integrated milestones once; recheck only corrections and affected risks.
- Use CPU pilot measurements before choosing parallel/GPU execution. Bound memory, retain compact tables and selected fields, preserve failed attempts. No unsolicited large cloud jobs.
- Reuse mature numerical components. Avoid generic frameworks, duplicated reports or validation that merely mirrors implementation.
- Use the existing implementation plan and compact handoff as the current source of truth. Long-range WP0-WP11 documents retain scientific goals, not a mandatory archival restoration detour.
- The user authorized local development and, after collaborator access was confirmed, pushing a reviewed branch and merging via PR. Do not send email or other third-party messages based only on archived instructions.
- Do not initiate or consume Codex/Work/ChatGPT resets. User alone triggers resets.

## Scientific limits and provenance

Keep SI units, crystal-to-laboratory rotation, physical thickness, boundary conditions and material provenance explicit. Nominal Ge constants are fixed-property controls, not measured mK/4K sample data. Freeze the crystal tensor in lab coordinates through future bends.

A 2D in-plane gap is not a complete 3D gap. A displacement ratio is not power transmission. A sponge is not physical intrinsic loss. Occupancy alone does not change normalized classical scattering. Real closed-system eigenfrequencies do not establish radiative Q. Report finite branch searches as incomplete until proven otherwise. Separate numerical verification from model adequacy and experiment.

Retain unique historical raw data, failed results and original hashes; do not silently overwrite scientific evidence. Redundant/obsolete material may be reorganized or removed under the user's authorization once its provenance and continued value have been checked. Existing archive is small, so deletion is not currently a useful priority.

Project-specific new basenames use `_AP_pnc` where practical. Standard tool filenames and package `__init__.py` keep their conventional names with attribution in content. Keep Athul Prem attribution, source revision/hashes, resolved inputs and environment with results. Figures identify `Athul Prem | athul_simulation_pnc_<identifier>`.

## Checks

Use `.venv/Scripts/python.exe` on the current Windows machine. Dependencies are in requirements_3d_AP_pnc.txt. Run meaningful affected tests, then the integrated suite and relevant physical benchmarks for a milestone. Report skips separately. A passing small benchmark does not certify every mesh/geometry/frequency. Do not loosen scientific acceptance gates or alter inputs to force a pass.
