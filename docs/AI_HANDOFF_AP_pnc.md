# Current development handoff

Author: Athul Prem | _AP_pnc

## Current core milestone

The user explicitly directed continued automatic development of the 26-hole simulator's core workflow on 2026-11-18, following the simplified plan. The one-time mainline assessment and cleanup are complete. Do not repeat them or extend the completed diagnostic series.

Current work connects the existing native frequency map to the workbench/headless worker: configure → check mesh → bounded frequency list → run → stop after the current point → resume saved original inputs → open saved map. Default new-point budget is one. Native map/CSV/report/checkpoints and scientific gates are reused. No new large device calculation, HPC authentication or campaign is required.

Root owns docs, Git, automation and integration on `codex/workbench-frequency-map`, based on PR #50 main `d86a1d4fbf56af3d1c7eca08af5787174307b605`. The inherited handoff/plan notes were the only tracked changes. `core_frequency_workflow` owns workbench_AP_pnc.py, tools/workbench_worker_AP_pnc.py and tools/map_scattering_AP_pnc.py; `core_frequency_tests` owns focused tests and independent review. Lead turn `01a1f088-231c-7ac2-957c-ed1108b160fc` is verified gpt-6-astra/ultra at 2026-11-18T21:20:16.328Z; workers use gpt-6.1-sol/ultra and distinct ownership.

Acceptance is the one portable tiny two-point public-worker case plus UI/protocol checks, then one integrated suite and CI. The tiny case must demonstrate accepted-point/shared-input reuse rather than repeat an archive solve. A separate retained pilot or scientific verifier is unnecessary. New full-device points and the lead-only proposal remain unqueued.

## Execution state

Hourly development remains ACTIVE in coordinating chat `01a1e7dd-949e-7ce0-b90f-9fad42ea564e`, with the existing cadence and failed-runs-only policy. Update that heartbeat to current completion state; do not create a duplicate schedule or ask whether to continue.

Native inventory at this activation found no SSH/SFTP/Python/numerical worker; all three old h80 locks and the port-count lock were empty. Lawrence was logged out at the earlier pause; the last actual remote queue observation was empty at 2026-11-18T03:33:07Z. Remote state has not been refreshed. Follow the [Lawrence runbook](LAWRENCE_HPC_AP_pnc.md) only when remote work is needed; root owns authentication, transfers and jobs.

Use `.venv/Scripts/python.exe` (Python 3.12.14). The previous app-update base-path repair is complete and recorded in results/runs/lead_phase_8875_AP_pnc/local_checks/runtime_repair_AP_pnc.json. Do not reinstall or repeat accepted simulations without a new reason. No Codex/Work/ChatGPT reset is permitted.

## Accepted results to reuse

The completed browser is available through **Open device results**, or `tools/browse_device_results_AP_pnc.py`; its local page is results/runs/saved_device_results_AP_pnc/report_AP_pnc.html. It joins h80 z6 at 0.1/0.8875/1 GHz, with 4/5/5 incident drives and 132 native coefficients. Exact public run paths and compact source identities are in configs/saved_device_dataset_AP_pnc.json. Original fields and nine finite controls remain linked in place. Do not rerun this data audit.

All three selected points pass their original-equation and physical-flux checks within the saved discretization. Low-band finite in-plane/thickness increments pass, but full spatial/frequency convergence is unestablished. The 0.8875 GHz weak-conversion and higher-frequency complex-response sensitivity gates fail. Retain these flags and raw physical planes/phases. Roots are frequency-local; the saved 0.8875/1 GHz signed endpoint relation is not continuous branch identity.

The saved regional energy and phase-only controls are complete and do not locate the cause of weak-conversion sensitivity. [Athul's incorporated comment](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/pull/48#issuecomment-6969077996) asks for a bounded new-control proposal before another solve. The exact lead-only proposal remains in the [plan](IMPLEMENTATION_PLAN_AP_pnc.md), unqueued. No phase subtraction, inferred numerical floor, converged spectrum or experimental-validation claim follows.

Existing accepted meshes/K/M/SAFE/solutions, unique failed attempts and evidence remain under results/ and verification/. Keep the plan's links as the evidence index; detailed history stays in Git and those ledgers rather than this handoff.

## Verification and integration

Baseline PR #50 is merged. Windows CI 44398928070 / job 135136307892 passed all 426 tests in 34.136 s without skips and all physical benchmarks; its full log is results/runs/saved_device_dataset_AP_pnc/local_checks/ci_windows_final_AP_pnc.log. The preceding local suite had one Windows symlink-privilege skip. Existing accepted numerical evidence is unchanged.

The frequency workflow is implemented and independently reviewed. All 24 affected tests pass in 27.851 s with no skips; a narrowly isolated browser-open fixture also passes. The one 405-DOF public-worker case checks preflight mesh byte/hash reuse, bounded first-point calculation, Stop received during that solve, original-config removal before resume, only the missing point solved next, and unchanged accepted summary/field bytes and mtimes. UI/protocol checks preserve exact 26-hole inputs and resource caps. No retained duplicate pilot or large device run was created. One integrated local suite ran 433 tests in 70.698 s: 432 passed, one Windows symlink-privilege skip. Its launcher took 74.219 s, with sampled process-tree RSS/private peaks 438935552/777691136 bytes within 1 GiB/180 s and a 512 MiB reserve; no guard fired. Logs are under results/runs/workbench_frequency_map_AP_pnc/local_checks/. Final-head CI and authorized PR integration remain. Direct browser local-file visual preview was previously disallowed by the browser security policy; do not retry through an alternate browser/server workaround. Qt/offscreen and public-worker behavior can be tested without that preview.

## Limits

Point-boundary Stop does not interrupt a factorization already running; display that boundary clearly. A forced process exit may require retry of the unfinished point, while accepted points remain protected. Native resume must retain exact input/source/environment checks, and imported read-only views must not become executable.

Nominal Ge is a fixed-property control. Affine faceted hole geometry, finite SAFE section roots, frequency-sorted Bloch samples and short reference acceptance retain their established limits. Full-device spatial/frequency convergence, general concurrent same-case scheduling and general crash recovery are not added by this interface milestone.
