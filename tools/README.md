# Integrity and archive tools

Author: **Athul Prem** | Project/file tag: **`_AP_pnc`**

[audit_repository_AP_pnc.py](audit_repository_AP_pnc.py) checks relocated-file hashes and reports archive availability; use `--list-missing` for details or `--require-complete-archive` for a strict gate. [archive_runtime_AP_pnc.py](archive_runtime_AP_pnc.py) resolves original paths using validated hashes and rejects unsafe paths/overwrites. It can reconstruct a complete recovered original tree, but it is not a numerical solver. Root [verify_AP_pnc.py](../verify_AP_pnc.py) checks the available saved smoke evidence. Root [run_AP_pnc.py](../run_AP_pnc.py) is a read-only readiness interface with real execution disabled pending recovery and adapter review.
