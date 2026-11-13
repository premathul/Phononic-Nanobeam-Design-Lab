# Development instructions

Research owner: **Athul Prem** | Project tag: **_AP_pnc**.

## Current direction

Read README.md, docs/AI_HANDOFF_AP_pnc.md, docs/IMPLEMENTATION_PLAN_AP_pnc.md and docs/PHYSICS_SPECIFICATION_AP_pnc.md. Current user authorization supersedes the historical handoff: architecture can change freely; prioritize the latest reliable physics and a usable configurable simulator. Full archive recovery is not a prerequisite for new development. Preserve actual evidence and label new implementation independently.

The active solver is `simulate_AP_pnc.py` and `src/phononic/`. Root `run_AP_pnc.py` remains a legacy readiness checker. Never confuse the two. Finite-body modes, Bloch dispersion and power-normalized uniform/perforated scattering are implemented. Two-hole power checks and a bounded 1 GHz complex-response mesh-pair check pass. The full 26-hole h200/h160 comparison failed both power and complex-response mesh criteria; h160/h130 passes power but still fails complex response. Independent SAFE port refinement also leaves phase accuracy unresolved. Reuse saved evidence before further solves, and keep those limits explicit before extending spectral or resonance claims.

The optional English Qt interface is `workbench_AP_pnc.py` (`Launch_Workbench_AP_pnc.cmd` on Windows); its headless worker calls the same solver and mesh-preflight APIs. UI dependencies are separate in `requirements_ui_AP_pnc.txt`. Keep mesh readiness, successful computation and scientific convergence distinct in the interface.

The workbench opens saved runs and compares scattering results through `tools/compare_scattering_AP_pnc.py` without new solves. Identical discretizations are repeatability checks, and failed input power checks cannot establish mesh-pair acceptance. The saved full-device/port follow-up is checked with `tools/check_device_followup_AP_pnc.py`; a successful integrity check does not mean the scientific convergence criteria passed.

The additional `layered` mesh uses `src/phononic/layered_AP_pnc.py` and `configs/perforated_layered_AP_pnc.json`. Its in-plane size and `thickness_layers` are separate controls; z layers also define the port thickness mesh. This remains a full 3D displacement model. `tools/diagnose_wave_resolution_AP_pnc.py` and the saved directional diagnostic establish sensitivity, not complete-device convergence. Keep the existing unstructured Gmsh path and its numerical evidence unchanged.

The tested two-hole 1 GHz thickness/port follow-up is complete. `configs/perforated_layered_reference_AP_pnc.json` selects h80/four layers/100 nm port width; final thickness and width increments and agreement with the saved unstructured reference pass 0.01. `tools/compare_uniform_reference_AP_pnc.py` compares saved uniform runs to the portable finer SAFE reference; coarse native-port phase alone can mislead. These bounded two-hole/600 nm-section checks do not certify the longer 26-hole/1000 nm-section device. Read the saved follow-up before new solves.

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
