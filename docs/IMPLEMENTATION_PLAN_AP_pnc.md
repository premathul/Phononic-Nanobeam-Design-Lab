# Three-dimensional straight-beam implementation plan

Author: Athul Prem | _AP_pnc

## Scope and deliverables

Develop the repository into a reproducible three-dimensional anisotropic elastic-wave simulator. The implemented workflow takes a configuration with explicit units, builds a 3D solid and mesh, applies the material tensor, and computes finite-beam eigenmodes, periodic dispersion or matched-port power scattering. The original 26-hole device is now integrated and has a one-frequency pilot. The two-hole 1 GHz reference meets its bounded power and complex-response mesh-pair checks. The next numerical milestone addresses full-device mesh sensitivity. A campaign spanning all seven temperatures, bends or ring variants is outside the current milestone.

The user authorized architectural changes and prioritized the latest reliable simulations. Historical recovery can proceed independently of the new 3D implementation. GitHub collaborator write access is confirmed. Push the development branch after integration checks and review, then merge through a pull request. Email delivery requires current user authorization.

## Baseline and reuse decisions

The local checkout began at `3aa7e12c0d033bc3494dbc8295ee2564c5743885`.

| Item | Decision | Rationale |
| --- | --- | --- |
| `data/geometry/hole_geometry_derived_AP_pnc.csv` and its original workbook | Reuse | The baseline supplies the 26-hole design. Accept the `position_nm` and `x_nm` column formats and convert to SI internally. |
| Latest available cubic-Ge configuration, 15 harmonic records and 4 GHz field | Retain as reference evidence | The saved results can be checked. Three-dimensional behavior and power transmission require independent verification. |
| Nominal stiffnesses C11/C12/C44 = 128.5/48.3/66.8 GPa and density 5323 kg/m³ | Reuse for fixed-property cases | Calibration to a specimen at 4 K or millikelvin temperatures remains unestablished. |
| Historical mesh, tensor and residual verification | Reuse the verification concepts | The historical mesh represents a two-dimensional solid. |
| `src/nanobeam_AP_pnc.py` | Retain as a legacy example | Its one-dimensional mass-spring chain illustrates dispersion. |
| Earlier isotropic FEM and later portable runner | Preserve in the archive | The isotropic solver uses different material assumptions. The portable runner is two-dimensional and lacks essential dependencies. |
| Historical figures and all 496 inventory entries | Keep historical recovery independent | The upload covers only part of the inventory. Recovery of unrelated figures does not improve the new model. |

The missing items include the original `handoff.py`, portable `numerics.py`, `config.json`, material and numerical-method notes, and the complete validation tables for the latest band-gap and cavity-mode results. The existing provenance files list the 223 unresolved entries, including 222 missing entries and one dependency with different bytes. The new 3D solver has independent provenance.

The initial Windows run encountered insufficient permission to create a symbolic link (WinError 1314). The test now records an explicit skip. The symbolic-link check remains unverified on that machine.

## Technical architecture

Use scikit-fem to assemble 3D vector finite elements, SciPy for sparse eigenvalue problems, Gmsh for circular-hole geometry and meshio for portable field output. Begin on CPU. Use structured tetrahedral meshes for small homogeneous-beam benchmarks and quadratic displacement elements by default. Use metres, pascals and kilograms internally.

| Module | Responsibility |
| --- | --- |
| `src/phononic/geometry_AP_pnc.py` | Parameterized straight beams, the original hole table, SI geometry and optional Gmsh meshes |
| `perforated_AP_pnc.py` | Port-compatible Gmsh meshes with independently controlled body, end-face and circle resolutions; exact CAD/affine-mesh volume diagnostics |
| `layered_AP_pnc.py` | Optional planar triangulation extruded into conforming affine tetrahedra, with independent in-plane size and explicit thickness layers |
| `material_AP_pnc.py` | Cubic crystal tensor, crystal-to-lab rotation and material provenance |
| `elasticity_AP_pnc.py` | Three-dimensional consistent mass and stiffness, eigenmodes and numerical residuals |
| `lead_modes_AP_pnc.py` | Bloch constraints on the two ends of a single mesh and guided-wave dispersion with free sidewalls |
| `port_modes_AP_pnc.py` | Semi-analytical finite-element (SAFE) section modes, signed power and all discrete propagating/evanescent roots |
| `scattering_AP_pnc.py` | Matched outgoing traction maps, incident drives, sparse harmonic solves and modal/direct-flux power checks |
| `workflow_AP_pnc.py` and `simulate_AP_pnc.py` | Configuration entry point, result identity, measured resources, tables, figures and reports |
| `mesh_checkpoint_AP_pnc.py` | Mesh-only preflight, exact tagged-mesh checkpoints and budget checks before assembly |
| Root `workbench_AP_pnc.py` and `tools/workbench_worker_AP_pnc.py` | Optional English Qt parameter interface and a headless structured-response worker using the existing solver |
| `tools/compare_scattering_AP_pnc.py` | Offline comparison of completed scattering runs with signed-power mode matching, phase alignment and separate input-quality/change criteria |

The workflow has a single launch command, editable configuration and a readable result page. Mesh-only preflight reports actual DOF counts and saves geometry for exact reuse after selecting a compute budget. The researcher-facing Qt interface connects presets, parameter edits, preflight, one background solve, process-tree cancellation, saved-run loading and offline scattering comparison. Reports expose both input power checks, power/complex changes and matched-channel phases; identical discretizations are repeatability checks. Its material and hole-placement metadata remain explicit; a successful run is still labeled unverified. Broader scientific editors can extend this interface after the corresponding models are established.

## Milestones and acceptance checks

1. **Three-dimensional mechanics.** Assemble the homogeneous beam with three displacement components, physical thickness, free surfaces and explicit clamps. Verify six free-body rigid modes, uniform strain energy, consistent mass, stiffness and mass symmetry, and agreement between independent dense and sparse solutions on a small problem. Record the measured residuals.
2. **Straight guided waves.** Apply `u(x+L)=exp(ikL)u(x)` using the same transformation for stiffness and mass. Check four zero modes at Γ, reciprocity between positive and negative wavenumbers, and the long-wavelength [100] extensional speed for free sidewalls. A finite free body has six rigid modes and a periodic straight cell has four at Γ. Report the number of sampled branches. Channel completeness and a complete gap require separate checks.
3. **Mesh and thickness.** Start with three systematically refined meshes at one thickness. Compare identified modes and changes before extending to 100, 200 and 1000 nm. Use an initial 0.5% frequency-change threshold and record any unconverged case. Verify the fixed-property null effect across temperature labels.
4. **Matched ports and power scattering.** Solve propagating and evanescent channels, normalize by energy flux, and assemble incident and outgoing boundary conditions. The homogeneous lossless control must give T near 1 and R near 0. Use an initial 0.01 threshold for error and changes under independent refinement. Establish this benchmark before running perforated devices. Report power directly from flux.
5. **Research extensions.** Study the 26-hole 3D control, thickness dependence, supported temperature-dependent properties, open bends, and eventually rings, loss and thermal transport. Absence of a gap or improvement is a valid result.

Use measured outcomes to select the next calculation. Review each integrated milestone and focus later checks on changes that affect the verified result.

The first mechanics and guided-wave milestone is merged. The matched-port implementation now passes the bounded 1 GHz, 600 × 200 nm uniform-guide benchmark for lengths of 1 and 1.5 µm. Its phase reference is a refined SAFE discretization, not an exact continuum solution. The initial coarse-mesh phase failures, subsequent refinements and final acceptance are retained in `verification/straight_scattering_*_AP_pnc.json`. Read the current handoff for measured results and limits before extending this benchmark.

CSV-defined perforation now runs through the public workflow, with source-byte checksums and the original coordinate offset preserved. The two-hole 1 GHz study meets the 0.01 absolute T/R sensitivity checks across body, circle, port and lead variations. Its original complex-coefficient change of 0.03162 failed the separate phase criterion, and that failed check remains in the evidence. Targeted follow-up at 60 and 50 nm body sizes reduced successive complex changes to 0.01367 and 0.00337; final T/R change is 0.000425. The last pair passes the unchanged 0.01 checks with fixed 48-segment holes, ports, nominal material and 1 GHz frequency. The portable four-level record and offline checker require no new simulations for review. This bounds the tested mesh-pair response, not the continuum error or a wider frequency band.

The original 26-hole, 56220-DOF pilot completed with five incident channels but is not mesh-converged. The bounded follow-up reused it and solved one 82056-DOF mesh at 1 GHz. It failed both 0.01 criteria: maximum power change 0.01136 and complex-coefficient change 0.42286. Body and hole-wall thickness discretization changed together; the latter went from one to two layers. A subsequent 130 nm, 96903-DOF mesh retains two wall layers and fixed hole faceting/ports. Its change from the 160 nm mesh passes power (0.002339) but fails complex response (0.137744). These three levels and their matched port bases remain saved; no existing solve needs repeating for comparison.

Independent SAFE-only ports with 99, 195 and 315 trace DOFs retain five propagating channels in each direction. The final pair changes k enough to give a 0.05638 rad phase indicator over the combined 3 micrometres of uniform leads, so port phase accuracy is unresolved. This indicator is not a full-device error bound and is not subtracted from the scattering coefficients. The completed follow-up and offline checker are `verification/device_convergence_followup_AP_pnc.*` and `tools/check_device_followup_AP_pnc.py`.

Directional controls are now implemented. A four-case 1.2 micrometre uniform-guide diagnostic independently varies axial spacing and thickness subdivision. At fixed ports, halving axial spacing decreases the maximum same-SAFE-reference phase error by about threefold; the finest case is still above 0.01. Its portable matrices and no-solve integrity test preserve the observed failures. The new layered mesh backend then applies separate footprint size and explicit z layers to through-hole geometries, with matching ports and unchanged polygonal holes. Layered z refinement changes both body and SAFE section; in-plane refinement at fixed layers holds the ports fixed.

The layered two-hole h105/h80/h60 study is complete at two z layers and fixed 135-DOF ports. Only its final pair passes the 0.01 complex-change criterion (0.006834). The h60 difference from the saved unstructured h50 reference is still 0.013431, and both layered uniform same-port reference checks fail. Thus the next bounded study must check thickness and port resolution independently; a stable in-plane pair is insufficient. Reuse all five layered public runs and the old reference without repeating them.

Use these controls and the saved SAFE results to choose a bounded full-device refinement before expanding a campaign. Hole faceting remains a separate geometric check. Define the selected observable and resource cap first; retain phase at the physical port planes. The saved-run comparator supports field-based matching across unequal P1/P2 port traces. Degenerate subspaces and changed channel counts need explicit treatment before those cases can use it. Do not rerun the unchanged straight-guide or two-hole campaigns.

## Resources and data

The initial machine measurements were about 15.31 GiB of physical RAM, 8 physical CPU cores and 16 logical processors, about 3.46 GiB of available RAM and about 130 GiB of free disk space. These values reflect the machine at the time of measurement. Begin with small tests of approximately 5000–10000 degrees of freedom or fewer. Run larger cases serially in the main workflow. The default 30000-degree-of-freedom limit bounds launch size. Safe capacity at that size still requires measurement. Record elapsed time, peak process memory and output size, then adjust based on the measurements.

Keep resolved configurations, source fingerprints, frequency and mode tables, selected fields and figures with the results. Assign separate run identities when inputs or sources change. Reuse verified complete runs and retain the status of unfinished attempts. Save selected fields to control output size during a sweep. Use the existing Git ignore rules for generated results and the local environment. Preserve the compact historical archive and its provenance, and organize access through versioned files and clear entry points.

## References and research scope

- QTCAD provides a reference for organizing geometry, mesh regions, materials, boundaries, solvers and results. Its strain tutorial uses prescribed strain as an electronic-structure input. The public documentation does not provide the elastic phononic-crystal solver needed here. [QTCAD workflow](https://docs.nanoacademic.com/qtcad/getting_started/workflow/), [strain tutorial](https://docs.nanoacademic.com/qtcad/tutorials/device/holes_strain/)
- The numerical implementation uses established examples and solver documentation. [scikit-fem 3D structural vibration](https://scikit-fem.readthedocs.io/en/stable/listofexamples.html#example-21-structural-vibration), [SciPy eigsh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html), [Gmsh periodic meshes](https://gmsh.info/doc/texinfo/gmsh.html#t18_003a-Periodic-meshes)
- The public material establishes context for cryogenic germanium and rare-event detection. The simulator's nanobeam acceptance checks come from the research goals documented in this repository. [Dongming Mei's University of South Dakota profile](https://www.usd.edu/research-and-faculty/faculty-and-staff/dongming-mei), [germanium athermal-phonon detector preprint](https://arxiv.org/abs/2610.29973)
- G4CMP is a reference for potential future athermal-phonon transport work. The current scope is continuum elastic-wave modes. Detector carrier and athermal-phonon transport require separate models. [G4CMP paper](https://arxiv.org/abs/2302.05998)
