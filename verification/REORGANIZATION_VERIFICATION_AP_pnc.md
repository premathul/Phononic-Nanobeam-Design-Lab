# Reorganization verification

Author: **Athul Prem** | Project/file tag: **`_AP_pnc`**
Baseline: `068cb8c92d179bfb2049f886278e95a0e7d9c709` | 12 November 2026.

## Outcome

All 199 uploaded files were preserved byte-for-byte at their new paths. The current available-evidence verifier passed, including 26 archived smoke-output hashes, 15 saved structure/frequency records, four saved operator comparisons and the available 4 GHz field arrays. All 12 active tests passed: four geometry/chain tests and eight archive/path/hash guard tests. No FEM simulation was rerun and no email was sent.

The archive remains incomplete: 273 of 496 expected entries are resolvable, including 84 exact-hash aliases; 223 are unresolved. This pre-existing upload condition is separately reported. A strict full-archive check correctly returns exit code 2. An attempted real launch correctly returns exit code 2 and creates no output directory. A dry run produces its blocker report with exit code 0; that code is not simulation success.

## Executed checks

| Command | Exit | Meaning |
| --- | ---: | --- |
| `python -B tools/audit_repository_AP_pnc.py` | 0 | Uploaded-byte preservation and explicit completeness report |
| `python -B verify_AP_pnc.py --fields --safety` | 0 | Available saved evidence and guard tests pass |
| `python -B -m unittest discover -s tests -p 'test*_AP_pnc.py' -v` | 0 | All 12 active tests pass |
| `python -B src/nanobeam_AP_pnc.py geometry data/examples/half_array_AP_pnc.csv --beam-width-nm 600` | 0 | Eight-hole example geometry check |
| `python -B src/nanobeam_AP_pnc.py chain --points 11` | 0 | Analytic illustrative chain output |
| `python -B run_AP_pnc.py --name hpge_reference --dry-run` | 0 | Read-only blocked-readiness report |
| `python -B run_AP_pnc.py --name hpge_reference` | 2 | Correctly blocked; no files/solver started |
| `python -B verify_AP_pnc.py --require-complete-archive` | 2 | Correctly reports incomplete historical archive |

The [machine-readable record](reorganization_checks_AP_pnc.json) contains actual outputs, timestamps and runtime metadata. The command list is scoped to repository organization and available evidence, not to the unimplemented 3D campaign.

## Preserved scientific distinctions

The available residual and source/sponge balance checks are algebraic/saved-artifact consistency evidence. They do not establish mesh convergence, complete port-power transmission, 3D behavior or specimen-calibrated temperature effects. The historical report of 333 harmonic and 60 cavity checks was preserved but could not be fully independently rechecked from the incomplete upload. Proposed tolerances and campaign counts remain labeled as plans.

## Layout and authorship

Project-specific filenames use `_AP_pnc` before the extension. Standard GitHub/tool filenames retain their names and attribution. The per-file index records ownership and hashes; historical bytes remain unchanged. Active Markdown links were checked against the organized tree; original documents retain historical links as provenance. JSON templates parse, execution remains disabled, and delivery status remains not prepared/not sent.
