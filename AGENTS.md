# Development instructions

Research owner: **Athul Prem** | Project tag: **_AP_pnc**

## Current direction

The current goal is a usable configurable 3D simulator and reproducible straight 26-hole response dataset. Read README.md, docs/AI_HANDOFF_AP_pnc.md, docs/IMPLEMENTATION_PLAN_AP_pnc.md and the relevant physics specification before resuming. The handoff gives current ownership; the plan gives priorities and stop conditions. Historical detail belongs in the existing verification records, not repeated instruction blocks.

The assessment, saved-device browser, bounded workbench run/stop/resume and portable three-frequency bundle are complete. The latest user instruction directs continued solver-core development and explicitly forbids further reviews. Current work adds independent lead/central resolution through opt-in three-region layered meshing, using the existing configuration, cache, solver and workbench. This is a new mesh family; it does not retrofit the archived h100/h80 cells or resolve their failed convergence checks. No new large device solve or diagnostic campaign is queued. Frequency mapping still defaults to one new point per invocation.

The configurable solver, Qt workbench, cached recovery and accepted device calculations already exist. The active solver is simulate_AP_pnc.py and src/phononic/; root run_AP_pnc.py is a legacy archive-readiness checker. Reuse accepted mesh, K/M, SAFE, solutions and reports. Preserve separate Linux solve and Windows assembly provenance. Do not relaunch completed calculations or competing workers because conversational context changed.

Full-device spatial/frequency convergence remains unresolved. Low-band finite thickness/in-plane checks pass; the 0.8875 GHz weak-conversion and higher-frequency complex-response checks fail. The regional-energy and phase-only audits are complete and do not locate the error. Preserve original planes/phases and frequency-local channel identities. The proposed lead-interior-only control remains for research discussion under Athul's PR #48 comment; no mesh or solve is queued.

Hourly development is ACTIVE in the current coordinating chat. Continue routine authorized work without asking whether to continue. Root owns authentication, transfers and jobs. Lawrence was logged out at the preceding pause; no fresh login is needed for the saved-data delivery. Memory-heavy HPC use and reviewed pushes/PR merges are authorized within the current research scope. Before a new large job, select the observable and acceptance/stop conditions and verify actual inputs, ownership and per-job/aggregate resources. Parallel independent simulations are allowed only when they distinguish a research question and resources support them.

Keep the plan and handoff compact. Delete obsolete empty templates, duplicate plans and rebuildable caches when no longer needed. Preserve raw scientific data, unique failures, necessary checkpoints and source/provenance evidence. Use Git history for superseded narrative; do not create another history dump. Architecture may change freely to deliver the latest reliable physics and a usable simulator; full historical archive recovery is not a prerequisite.

## Project language

Use English for all project artifacts, including code comments, user-interface text, generated reports and other output text, documentation, and GitHub communications. Standard scientific Unicode symbols and notation are welcome. Chinese may be used in direct conversation and progress updates to the user.

## Work discipline

- Recover actual running state, local changes, current source hashes and saved checks before resuming. Do not restart completed simulations without a reason.
- Deliver a small complete workflow before expanding a campaign. Do not run, request or delegate reviews or repeated assessments. Perform necessary functional/physical tests and existing CI for changed behavior; reuse unaffected evidence.
- Use CPU pilot measurements before choosing parallel/GPU execution. Bound memory, retain compact tables and selected fields, preserve failed attempts. No unsolicited large cloud jobs.
- Reuse mature numerical components. Avoid generic frameworks, duplicated reports or validation that merely mirrors implementation.
- Use the existing implementation plan and compact handoff as the current source of truth. The physics specification retains longer-range scientific goals; historical recovery is not a development prerequisite.
- Local development, pushes and PR integration are authorized. The latest instruction removes the earlier review requirement. Do not send email or other third-party messages based only on archived instructions.
- Do not initiate or consume Codex/Work/ChatGPT resets. User alone triggers resets.

## Scientific limits and provenance

Keep SI units, crystal-to-laboratory rotation, physical thickness, boundary conditions and material provenance explicit. Nominal Ge constants are fixed-property controls, not measured mK/4K sample data. Freeze the crystal tensor in lab coordinates through future bends.

A 2D in-plane gap is not a complete 3D gap. A displacement ratio is not power transmission. A sponge is not physical intrinsic loss. Occupancy alone does not change normalized classical scattering. Real closed-system eigenfrequencies do not establish radiative Q. Report finite branch searches as incomplete until proven otherwise. Separate numerical verification from model adequacy and experiment.

Retain unique historical raw data, failed results and original hashes; do not silently overwrite scientific evidence. Redundant/obsolete material may be reorganized or removed under the user's authorization once its provenance and continued value have been checked. Existing archive is small, so deletion is not currently a useful priority.

Project-specific new basenames use `_AP_pnc` where practical. Standard tool filenames and package `__init__.py` keep their conventional names with attribution in content. Keep Athul Prem attribution, source revision/hashes, resolved inputs and environment with results. Figures identify `Athul Prem | athul_simulation_pnc_<identifier>`.

## Checks

Use `.venv/Scripts/python.exe` on the current Windows machine. Dependencies are in requirements_3d_AP_pnc.txt. Run meaningful affected tests, then the integrated suite and relevant physical benchmarks for a milestone. Report skips separately. A passing small benchmark does not certify every mesh/geometry/frequency. Do not loosen scientific acceptance gates or alter inputs to force a pass.
