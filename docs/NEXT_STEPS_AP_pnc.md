# Ordered research tasks and acceptance gates

Author: **Athul Prem** | Project/file tag: **`_AP_pnc`**
Status at reorganization: **Planning and partial-archive audit; no new 3D calculations executed.**

## Execution order

The continuing AI should work through the following packages in order. Independent literature work and geometry design may proceed while archive recovery is blocked, but missing baseline evidence must stay explicit. A package is complete only when its stated deliverables and gate are recorded with commands, source revision and evidence paths.

### WP0. Recover and establish the baseline

Run the current audit, saved-smoke verifier and lightweight tests. Consult `provenance/missing_archive_files_AP_pnc.json`; 222 entries are absent and the uploaded dependency file differs from its historical expected hash. The 84 duplicate-hash aliases already map to existing verified bytes, without duplicating large files.

Obtain exact original files from Athul's full project/backup when available. Prioritize `handoff.py`, the portable template's `numerics.py`, `config.json`, methods/material notes and three postprocessors, then the original provenance hash map and latest-run validation/band/cavity tables. Match SHA-256 before admitting a recovered file. Preserve the original missing list as a dated snapshot, update the active catalogue with explicit recovery provenance and review the execution adapter before enabling a real launch. Never recreate missing numerical code and call it recovered original code.

**Deliver:** recovery ledger, complete-archive audit, reviewed runnable entrypoint, source/environment record and the smallest fresh smoke comparison once executable. **Gate:** hash consistency and a documented fresh baseline match. A source reconstruction, if required, is a new solver version with independent tests.

### WP1. Define material and admissible regimes

Read primary sources and applicable corrections. Build `material_registry_AP_pnc.json` and `material_provenance_AP_pnc.csv`, with units, intervals, uncertainties and interpolation choices. Obtain actual orientation, thickness metrology and purity/isotope information where available. Separate unknowns from assumed nominal properties. Verify positive elastic energy, tensor rotation and mass/dilation consistency.

Assign each temperature case to a documented policy. Treat 0.02/0.04/1 K as fixed-property or justified low-temperature-model cases until adequate evidence supports more. Do not fit arbitrary dissipation to make a plot look realistic.

**Deliver:** sourced registry, property/uncertainty plots, validity table and model-admission tests. **Gate:** every physical input traceable and every unsupported case labeled or rejected.

### WP2. Implement a reusable 3D geometry and configuration system

Build homogeneous, periodic and defect beams using the original 26-hole dimensions. Extrude 100/200/1000 nm solids with free top, bottom and hole walls. Parameterize width, pitch, hole diameter/taper, leads and supports. Keep the initial planar pattern identical across thicknesses. Reject overlaps, negative ligaments, bad volumes and incompatible ports before a solve.

Implement strict configuration/schema validation using the supplied templates as design inputs. Reject missing units, unsupported material policy, invalid rotation and unresolved resource limits. Assign separate full-run and mechanical-operator hashes. Create atomic checkpoints and per-run locks; source or scientific-input changes require a new identity.

**Deliver:** geometry factory, valid small example, dimensioned views, checksums, schema and restart-equivalence check. **Gate:** exact geometry preserved under independent mesh realizations and deterministic resolved configuration.

### WP3. Verify mechanics and homogeneous modes

Implement 3D stiffness/mass assembly and suitable solvers. Check rigid-body behavior, uniform-strain patches, units, symmetry and positive strain energy. Compare dense/sparse or independent small operators where practical. Verify bulk Christoffel speeds and long-wave beam/rod limits in their valid regimes. Identify extensional, both flexural, torsional and mixed modes at each required thickness.

**Deliver:** small benchmark suite, residual tables, dispersion/mode profiles and convergence plots. **Gate:** reference benchmarks and stable mode identity; no arbitrary mode-count cap treated as completeness.

### WP4. Calibrate ports, energy flow and scattering

Find all propagating lead modes; refine evanescent content. Normalize incident power and compute full outgoing channel amplitudes with the appropriate flux form. Test a known traveling wave's flux sign, homogeneous unit transmission and reciprocity of corresponding channels. Vary lead lengths, monitors, absorber settings and modal truncation independently of mesh.

**Deliver:** `mode_inventory_AP_pnc.csv`, `scattering_channels_AP_pnc.csv`, port tests and energy budgets. **Gate:** initial absolute power/T/R targets below 0.01 with independent convergence. Resolve the physical causes of the historical 2.8 GHz sponge and 5.2 GHz mesh failures in targeted new controls before trusting broad patterned spectra.

### WP5. Produce the straight-device atlas

Start with one supported reference-temperature policy and the three thicknesses. Compare homogeneous, periodic and finite-defect devices using matched ports. Survey 0.1-6 GHz with 121 initial points, insert cutoffs/features, and adaptively refine. Establish whether any gap is complete across the admitted 3D channels or limited to a mode family.

**Deliver:** mode/cutoff atlas, power T/R spectra, conversion matrices, selected 3D fields and thickness-frequency maps. **Gate:** representative spatial, modal, boundary and frequency convergence; no claimed suppression below the validated floor.

### WP6. Separate temperature effects

Deliberately rerun at least two frozen-mechanical-property temperatures as a null test. Then implement sourced material-only variation, and only later free contraction or support-induced prestress. Track modes across temperature using fields/subspaces. At supported conditions compare the same geometry and frequency axes across all three thicknesses.

**Deliver:** temperature/thickness maps; material-versus-occupation decomposition; edge, cutoff and mode trajectories; admitted/rejected temperature table. **Gate:** every nonzero mechanical thermal shift has an identified input cause and exceeds relevant uncertainty. After the null test, explicitly reuse identical mechanical spectra rather than redundantly solving each occupation label.

### WP7. Compare open circular-pattern devices

At one supported reference temperature screen `R/w = 5, 10, 20, 40` and the straight control, for each thickness. Match path length, cross-section, local pattern and material policy as closely as geometry allows. Specify lead transitions and reject infeasible arcs. Keep the single-crystal tensor fixed globally.

**Deliver:** circular/straight geometry table, curvature-T/R/conversion maps, selected fields and large-radius consistency test. **Gate:** observed differences survive independent numerical refinement and matched-control checks. Keep a representative non-improving case when selecting finalists.

### WP8. Study ring topologies and dimension sensitivity

Treat a closed ring and bus-coupled ring as separate boundary-value problems. Specify circumference/pitch closure, hole count, taper/defect and coupling gap/supports. For a closed ring enumerate mode families and splitting; for a coupled ring compute through/drop response and independently converged linewidths. Do not derive radiative Q from a closed lossless real-eigenvalue problem.

Screen width, diameter, pitch and taper around reference factors 0.9/1.0/1.1 where feasible, then test selected interactions. These factors are exploratory sensitivities, not established fabrication tolerances.

**Deliver:** ring modes, coupling study and dimension-sensitivity tables. **Gate:** mode identity, topology-specific controls and an uncertainty-resolved conclusion.

### WP9. Add loss/thermal transport only if justified

Select supported loss models and verify passivity/sign conventions against simple attenuation benchmarks. Compare device lengths with justified mean-free-path estimates and distinguish coherent from incoherent regimes. Use multiple matched lengths to separate input reflection from propagation decay. If computing a Landauer-type conductance, preserve integration units and distinguish the finite-band contribution from total conductance or bulk conductivity.

**Deliver:** optional attenuation/lifetime or band-limited thermal analysis, with validation intervals and uncertainties. **Gate:** physically sourced model and separate convergence. If unsupported, retain the elastic results and explain this extension's block.

### WP10. Quantify uncertainty and synthesize findings

Tighten finalists' meshes, port/modal bases and spectral resolution. Vary documented material/dimensional uncertainties and assess the stability of each claim. Compare an independent implementation or external benchmark where possible. A numerical verification result is not experimental validation.

**Deliver:** claim-by-claim ledger, uncertainty bands, limitations and a resolved or explicitly unresolved design ranking. **Gate:** conclusions are supported by accessible data and error estimates; failed points remain listed.

### WP11. Produce and deliver the final research report

Write a full scientific report and reproducibility package using `reports/REPORT_REQUIREMENTS_AP_pnc.md`. Record actual completed counts of geometries, thicknesses, temperatures, modes, frequencies, failed points and rejected cases. New figures show Athul Prem at the top and use the required identifiers. Send the final report to **athul.prem@coyotes.usd.edu** using authenticated email when available; record the returned delivery evidence. If unable to send, save the complete report and draft and state pending delivery.

## Planned campaign size

| Campaign | Planned spectra | Initial frequency configurations | Additional work |
| --- | ---: | ---: | --- |
| Straight periodic + defect: 3 thicknesses x 7 temperatures x 2 | 42 | 5082 | Homogeneous controls, mode excitations, supported policies, refinements |
| Open bends: 4 nonzero radii x 3 thicknesses at one temperature | 12 | 1452 | Matched straight controls and selected temperature extensions |
| Initial combined design | 54 | 6534 | Not a count of executed FEM solves |

Each frequency may require multiple incident-channel right-hand sides. Reuse a factorization only when the operator is identical. Identical mechanical fixed-property cases may reuse spectra after the null check. An optional 8 GHz extension, ring campaigns and factorial dimension sweeps need their own cost/validity decisions.

Before a broad sweep, pilot the largest representative thickness/channel case and record peak memory, factorization time, per-channel solve cost and disk growth. Supply explicit RAM, disk and elapsed-time limits. Keep compact tables at every point and full fields only where they support a claim or diagnose a failure. Stop cleanly before exceeding the budget and preserve resumable state. Do not assume Athul's laptop or an unspecified cloud machine can support the full matrix.

## Required final tables

1. Material sources, uncertainties and valid intervals.
2. Geometry/dimensions/thickness/curvature and actual admitted campaign matrix.
3. Mode inventory, cutoffs, polarization and velocities.
4. Channel scattering coefficients and power T/R/absorption budgets.
5. Mesh, wavevector, modal, boundary and frequency convergence.
6. Temperature-policy decomposition and supported shifts.
7. Matched straight/bent/ring comparisons and failed/rejected cases.
8. Dimension/material sensitivity and uncertainty-resolved rankings.
9. Runtime, memory, storage and checkpoint behavior.
10. Claim-by-claim evidence and limits of interpretation.

## Required figure families when data support them

Dimensioned 3D geometry and port views; material curves; dispersion/cutoffs with mode character; representative 3D modes; T/R and conversion spectra; temperature-frequency and thickness-frequency maps; curvature maps; displacement/strain/flux views at scientifically chosen points; convergence/absorber tests; null-temperature comparison; ring-mode/coupling plots; uncertainty bands; design trade-offs; compute scaling. Avoid redundant figures that answer no additional question. Scientific plot colors may be used; report prose should remain black on white.
