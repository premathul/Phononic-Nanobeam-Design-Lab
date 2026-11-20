# Research and execution tools

Author: **Athul Prem** | Project/file tag: **`_AP_pnc`**

[Project home](../README.md) · [Environment](../docs/ENVIRONMENT_AP_pnc.md) · [Solver source](../src/README.md)

## Read and reproduce saved evidence

| Tool | Inputs and effect |
| --- | --- |
| [Saved-device browser](browse_device_results_AP_pnc.py) | Python standard library. Generate the offline reader from the pinned compact JSON/CSV bundle. Add `--open` to view it. [Data coverage](../data/README.md#portable-saved-device-data) |
| [Publication figure generator](plot_saved_device_figures_AP_pnc.py) | Matplotlib and tracked figure inputs. Regenerate vector PDFs, PNG previews and tables from the repository. [Commands and sources](../figures/README.md#reproduce-the-figures) |
| [Saved scattering comparison](compare_scattering_AP_pnc.py) and [comparison inspector](inspect_scattering_comparison_AP_pnc.py) | Analyze compatible saved responses and report aligned power and complex increments. Use each comparison's original planes and source requirements. [Verification records](../verification/README.md) |
| [Archive frequency-map importer](import_lawrence_frequency_map_AP_pnc.py) | Read the original Lawrence 1 GHz anchor and recorded hashes. The export is read-only, with other requested frequencies pending. |
| [Endpoint projection comparison](compare_snapshot_midpoint_AP_pnc.py) | Replay the stored midpoint predictions and their residual failures against the compact thin-device data. The [projection runner](run_snapshot_midpoint_AP_pnc.py) instead needs the native operators and full fields. |

## Prepare, compute and resume

Use [map_scattering_AP_pnc.py](map_scattering_AP_pnc.py) with the 3D numerical environment, resolved configuration and explicit increasing frequencies in GHz. `--max-new-points 0` stages shared mesh, stiffness, mass and port operators. Repeat with `--max-new-points 1` and the same identities to attempt one pending point. Failed attempts count toward the limit. `--mesh-cache-root` reuses a matching preflight mesh. Small guide example:

```powershell
.\.venv\Scripts\python.exe tools/map_scattering_AP_pnc.py --config configs/straight_scattering_AP_pnc.json --frequencies-ghz 0.3 --output results/runs/straight_guide_map_AP_pnc --max-new-points 1
```

A native map resumes the saved configuration, frequencies, source and environment. [seed_frequency_map_AP_pnc.py](seed_frequency_map_AP_pnc.py) with `--from-map`, a separate `--output` and explicit frequencies reuses a verified donor's preparation for new samples. The [preparation transfer tool](transfer_frequency_preparation_AP_pnc.py) binds authorized platform transfer to the original source and matrix identities. See the [Lawrence runbook](../docs/LAWRENCE_HPC_AP_pnc.md) for operator staging and recovery.

In the workbench, **Check mesh** precedes **Run frequency map**. **Run from saved mesh…** seeds from the exact native donor. **Stop after current point** lets the current attempt finish. **Resume map…** uses saved inputs, and **Open saved map…** opens a published map without solving or editing the current inputs. A native retry preserves completed points and records pending or failed points. The map's calculation, original-equation, physical-power, coherent and uniform-guide statuses have separate meanings. See the [claim and evidence contract](../docs/RESEARCH_STATUS_AP_pnc.md) and the [configuration guide](../configs/README.md).

For an individual iterative study, the [simulator](../simulate_AP_pnc.py) supports `--checkpoint --wall-seconds 1800` and `--resume PATH_TO_SAVED_ATTEMPT --wall-seconds 1800`. The original equation and input-identity checks govern recovery.

## Historical archive tools

The frozen Lawrence port-inventory producer is retained byte for byte because the archived execution chain binds its exact source. Its existing CSV and provenance JSON remain the authority for signed per-sample channel counts; the inventory is port-only evidence. Historical replay tools with fixed producer hashes require the recorded source checkout as well as the original field archive. Current report formatting does not change those producer identities or grant cache compatibility. The maintained reader and comparison writers record their own current source separately.

[audit_repository_AP_pnc.py](audit_repository_AP_pnc.py) verifies uploaded-file relocation and reports missing originals. Use `--list-missing` for the inventory or `--require-complete-archive` for the strict completeness check. [archive_runtime_AP_pnc.py](archive_runtime_AP_pnc.py) resolves original paths against recorded hashes and guards reconstruction. Root [verify_AP_pnc.py](../verify_AP_pnc.py) checks the available saved smoke evidence. Root [run_AP_pnc.py](../run_AP_pnc.py) reports legacy readiness and has real execution disabled. [Historical archive and recovery](../docs/EVIDENCE_AND_RECOVERY_AP_pnc.md)
