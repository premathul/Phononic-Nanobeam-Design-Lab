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
| `material_AP_pnc.py` | Cubic crystal tensor, crystal-to-lab rotation and material provenance |
| `elasticity_AP_pnc.py` | Three-dimensional consistent mass and stiffness, eigenmodes and numerical residuals |
| `lead_modes_AP_pnc.py` | Bloch constraints on the two ends of a single mesh and guided-wave dispersion with free sidewalls |
| `port_modes_AP_pnc.py` | Semi-analytical finite-element (SAFE) section modes, signed power and all discrete propagating/evanescent roots |
| `scattering_AP_pnc.py` | Matched outgoing traction maps, incident drives, sparse harmonic solves and modal/direct-flux power checks |
| `workflow_AP_pnc.py` and `simulate_AP_pnc.py` | Configuration entry point, result identity, measured resources, tables, figures and reports |

Begin with a single launch command, editable configuration and a readable result page. Add a researcher-facing parameter interface after the workflow is established.

## Milestones and acceptance checks

1. **Three-dimensional mechanics.** Assemble the homogeneous beam with three displacement components, physical thickness, free surfaces and explicit clamps. Verify six free-body rigid modes, uniform strain energy, consistent mass, stiffness and mass symmetry, and agreement between independent dense and sparse solutions on a small problem. Record the measured residuals.
2. **Straight guided waves.** Apply `u(x+L)=exp(ikL)u(x)` using the same transformation for stiffness and mass. Check four zero modes at Γ, reciprocity between positive and negative wavenumbers, and the long-wavelength [100] extensional speed for free sidewalls. A finite free body has six rigid modes and a periodic straight cell has four at Γ. Report the number of sampled branches. Channel completeness and a complete gap require separate checks.
3. **Mesh and thickness.** Start with three systematically refined meshes at one thickness. Compare identified modes and changes before extending to 100, 200 and 1000 nm. Use an initial 0.5% frequency-change threshold and record any unconverged case. Verify the fixed-property null effect across temperature labels.
4. **Matched ports and power scattering.** Solve propagating and evanescent channels, normalize by energy flux, and assemble incident and outgoing boundary conditions. The homogeneous lossless control must give T near 1 and R near 0. Use an initial 0.01 threshold for error and changes under independent refinement. Establish this benchmark before running perforated devices. Report power directly from flux.
5. **Research extensions.** Study the 26-hole 3D control, thickness dependence, supported temperature-dependent properties, open bends, and eventually rings, loss and thermal transport. Absence of a gap or improvement is a valid result.

Use measured outcomes to select the next calculation. Review each integrated milestone and focus later checks on changes that affect the verified result.

The first mechanics and guided-wave milestone is merged. The matched-port implementation now passes the bounded 1 GHz, 600 × 200 nm uniform-guide benchmark for lengths of 1 and 1.5 µm. Its phase reference is a refined SAFE discretization, not an exact continuum solution. The initial coarse-mesh phase failures, subsequent refinements and final acceptance are retained in `verification/straight_scattering_*_AP_pnc.json`. Read the current handoff for measured results and limits before extending this benchmark.

CSV-defined perforation now runs through the public workflow, with source-byte checksums and the original coordinate offset preserved. The two-hole 1 GHz study meets the 0.01 absolute T/R sensitivity checks across body, circle, port and lead variations. Its original complex-coefficient change of 0.03162 failed the separate phase criterion, and that failed check remains in the evidence. Targeted follow-up at 60 and 50 nm body sizes reduced successive complex changes to 0.01367 and 0.00337; final T/R change is 0.000425. The last pair passes the unchanged 0.01 checks with fixed 48-segment holes, ports, nominal material and 1 GHz frequency. The portable four-level record and offline checker require no new simulations for review. This bounds the tested mesh-pair response, not the continuum error or a wider frequency band.

The original 26-hole, 56220-DOF pilot completed with five incident channels but is not mesh-converged. The next complete workflow reuses that pilot, measures a finer mesh before allocating a solve, compares the same five left-incident channels at 1 GHz, and records power and complex-response changes. Port and circle refinements remain separate checks. Finish this bounded comparison before extending a frequency campaign; do not rerun the unchanged straight-guide or two-hole campaigns.

## Resources and data

The initial machine measurements were about 15.31 GiB of physical RAM, 8 physical CPU cores and 16 logical processors, about 3.46 GiB of available RAM and about 130 GiB of free disk space. These values reflect the machine at the time of measurement. Begin with small tests of approximately 5000–10000 degrees of freedom or fewer. Run larger cases serially in the main workflow. The default 30000-degree-of-freedom limit bounds launch size. Safe capacity at that size still requires measurement. Record elapsed time, peak process memory and output size, then adjust based on the measurements.

Keep resolved configurations, source fingerprints, frequency and mode tables, selected fields and figures with the results. Assign separate run identities when inputs or sources change. Reuse verified complete runs and retain the status of unfinished attempts. Save selected fields to control output size during a sweep. Use the existing Git ignore rules for generated results and the local environment. Preserve the compact historical archive and its provenance, and organize access through versioned files and clear entry points.

## References and research scope

- QTCAD provides a reference for organizing geometry, mesh regions, materials, boundaries, solvers and results. Its strain tutorial uses prescribed strain as an electronic-structure input. The public documentation does not provide the elastic phononic-crystal solver needed here. [QTCAD workflow](https://docs.nanoacademic.com/qtcad/getting_started/workflow/), [strain tutorial](https://docs.nanoacademic.com/qtcad/tutorials/device/holes_strain/)
- The numerical implementation uses established examples and solver documentation. [scikit-fem 3D structural vibration](https://scikit-fem.readthedocs.io/en/stable/listofexamples.html#example-21-structural-vibration), [SciPy eigsh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html), [Gmsh periodic meshes](https://gmsh.info/doc/texinfo/gmsh.html#t18_003a-Periodic-meshes)
- The public material establishes context for cryogenic germanium and rare-event detection. The simulator's nanobeam acceptance checks come from the research goals documented in this repository. [Dongming Mei's University of South Dakota profile](https://www.usd.edu/research-and-faculty/faculty-and-staff/dongming-mei), [germanium athermal-phonon detector preprint](https://arxiv.org/abs/2610.29973)
- G4CMP is a reference for potential future athermal-phonon transport work. The current scope is continuum elastic-wave modes. Detector carrier and athermal-phonon transport require separate models. [G4CMP paper](https://arxiv.org/abs/2302.05998)
