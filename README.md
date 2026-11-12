# Phononic Nanobeam Design Lab

**Author and research owner: Athul Prem**
**Project/file identifier: `_AP_pnc`**
**Final research-report recipient: athul.prem@coyotes.usd.edu**

## Research purpose

Develop a reproducible, three-dimensional elastic-wave research system for nominally pure high-purity germanium (HPGe). Determine how physical thickness, perforation geometry, curvature and temperature influence propagating phonon modes, reflection, mode conversion and localization. The central deliverable is a **mode-resolved propagation atlas with numerical and material uncertainties**, followed by a defensible comparison of straight, bent and circular devices.

The required thicknesses are **100 nm, 200 nm and 1000 nm (1 micrometre)**. Compare the same in-plane geometry across thicknesses and the same reference device across temperatures. A circular arrangement means a curved/annular propagation path; the historical straight beam already contains circular holes. No winning geometry, phononic gap, improved confinement or publication outcome is assumed in advance.

This folder-based layout implements Athul's current request and supersedes the previous flat-folder layout. The supplied [12-page continuation brief](docs/references/Phononic_Research_Continuation_AP_pnc.pdf) remains the scientific starting point; its old paths and conditional flat-layout guidance are superseded by this README and the [path manifest](provenance/relocation_manifest_AP_pnc.json).

## Current status: read before running

**The full archived FEM workflow is incomplete in the GitHub upload. The new 3D HPGe campaign is specified, but has not been implemented or run.**

All **199 files** uploaded at baseline commit `068cb8c92d179bfb2049f886278e95a0e7d9c709` are preserved byte-for-byte at documented new paths. The older manifest describes 496 archive entries: 189 match directly, 84 can be mapped to exact duplicate bytes elsewhere in the upload, and **223 remain unresolved** (222 missing entries and one changed dependency file). Missing components include the archived launcher, portable numerical module/configuration and full latest-run validation tables. They cannot be repaired by renaming files or installing NumPy.

What works now is the repository integrity audit, the available saved-smoke consistency checks, elementary geometry checks and the illustrative analytic chain. `run_AP_pnc.py --dry-run` reports readiness and blockers. A real FEM launch remains blocked until the missing original evidence and runner are recovered and the execution adapter is reviewed and enabled. This reorganization does not add new simulated physics results.

See [evidence and recovery status](docs/EVIDENCE_AND_RECOVERY_AP_pnc.md) for exact distinctions between recomputed checks, archived reports and proposed work.

## Folder guide

| Folder/file | Purpose |
| --- | --- |
| [docs/](docs/README.md) | Detailed physics programme, ordered tasks, environment, primary references and AI handoff |
| [configs/](configs/README.md) | Explicitly non-executable campaign and material-registry templates |
| [data/](data/README.md) | Original 26-hole workbook/CSV, geometry, meshes and small examples |
| [src/](src/README.md) | Active lightweight geometry/analytic code; no 3D FEM solver yet |
| [tools/](tools/README.md) | Hash, path, availability and saved-evidence checks |
| [tests/](tests/README.md) | Scientific unit checks and migration/integrity guard tests |
| [results/](results/README.md) | Available historical numerical tables, fields, inputs and logs, separated by lineage |
| [figures/](figures/README.md) | Existing figures grouped by archive/run; historical pixels unchanged |
| [archive/](archive/README.md) | Original solvers, adapters, documentation and older attempts, retained for provenance |
| [provenance/](provenance/README.md) | Original hashes, old-to-new paths, per-file attribution and missing-file catalogue |
| [verification/](verification/README.md) | Historical verification reports and checks performed during this reorganization |
| [reports/](reports/README.md) | Final-report template and required email-delivery procedure |
| [AGENTS.md](AGENTS.md) | Instructions the continuing AI must read first |

## Runnable checks

Use Python 3.11 or newer with NumPy. From the repository root:

```bash
python -m pip install -r requirements.txt
python tools/audit_repository_AP_pnc.py
python verify_AP_pnc.py --fields --safety
python -m unittest discover -s tests -p 'test*_AP_pnc.py' -v
python src/nanobeam_AP_pnc.py geometry data/examples/half_array_AP_pnc.csv --beam-width-nm 600
python src/nanobeam_AP_pnc.py chain --points 11
python run_AP_pnc.py --name hpge_reference --dry-run
```

The example CSV has eight holes and is a small interface test; it is not the 26-hole research geometry. The analytic chain demonstrates longitudinal mass-spring dispersion and is not a model of the perforated germanium beam. The default verifier's successful result means the **available evidence** passes its stated checks; inspect the separate archive-completeness status. The strict recovery gate is:

```bash
python verify_AP_pnc.py --require-complete-archive
```

That command is expected to fail until the missing archive is recovered. The [environment guide](docs/ENVIRONMENT_AP_pnc.md) explains the lightweight dependencies and historical FEM environment separately.

## What the existing evidence supports

| Evidence | Present interpretation |
| --- | --- |
| 26 saved-smoke outputs match their recorded hashes | Available archived outputs are internally consistent and were preserved |
| Five saved frequencies, three structures: 15 harmonic records | Saved residuals and displacement-ratio arithmetic can be checked without solving again |
| Defect/reference response at 4 GHz: approximately -42.2487 dB | A displacement-amplitude proxy, not calibrated power transmission |
| Reported latest in-plane gap: 2.953106-3.062947 GHz | Historical 2D cubic-Ge result; complete supporting latest-run tables are absent from this upload |
| Reported candidate: approximately 3.04210 GHz | Historical moderately centered eigenmode; not a demonstrated 3D HPGe cavity or measured device |
| Reported 2.8 GHz sponge span: 3.26848 dB; 5.2 GHz refinement change: -9.05400 dB | Unresolved boundary and mesh sensitivity; retain these failure reports |

The uploaded [smoke configuration](results/archive_smoke/run_new_verified_smoke/config_AP_pnc.json) explicitly uses an assumed (001)/[100] orientation, nominal cubic constants, plane stress, unit thickness and a numerical sponge. Its temperature changes occupation rather than the elastic operator. The historical report of 333 harmonic checks and 60 cavity residual checks is preserved, but this incomplete upload cannot independently recheck all of them. A small algebraic residual does not establish physical convergence.

## Detailed physics endeavour

### 1. Define the material before interpreting temperature

Use crystalline Ge without intentional doping or added material layers. Record impurity/resistivity data, isotope composition, defects, crystal axes and the source of each mechanical parameter. Without specimen records, call the material **nominal pure HPGe**. High chemical purity does not establish isotopic enrichment or eliminate surface/boundary scattering.

Build a registry for `C11(T)`, `C12(T)`, `C44(T)`, density, thermal expansion and any physical loss law. Each record needs units, a source actually consulted, measurement context, uncertainty, supported temperature/frequency intervals and an interpolation policy. The nominal constants in the historical files are not verified millikelvin or specimen-specific inputs. The [reference list](docs/REFERENCES_AP_pnc.md) identifies starting literature and its limits; no cited curve has been silently installed.

The proposed temperature grid is **0.02, 0.04, 1, 4, 20, 77 and 300 K**. Admit a quantitative material-dependent case only where its property model is supported. Unsupported points may be fixed-property controls or labeled sensitivity scenarios. Separate three questions: occupation changes with mechanics frozen; material-property changes at fixed dimensions; and free contraction or mechanically constrained thermal stress. Repeat two frozen-property temperatures as a null experiment: normalized elastic transmission and eigenfrequencies must agree within numerical error.

### 2. Solve actual three-dimensional anisotropic elasticity

For displacement amplitude `u` and the `exp(-i omega t)` convention, use

$$
\epsilon_{kl}=\tfrac12(\partial_k u_l+\partial_l u_k),\qquad
\sigma_{ij}=C_{ijkl}\epsilon_{kl},\qquad
-\partial_j\sigma_{ij}=\rho\omega^2u_i.
$$

Construct physical 3D volumes with traction-free top, bottom and hole walls except at explicitly modeled ports/supports. Begin with a homogeneous lead, periodic perforated guide and finite defect beam. Preserve the baseline 26-hole pattern and the provisional 1000 nm beam width/1500 nm end leads for the controlled first comparison. Thickness must enter the solid and displacement field; changing a scalar multiplier in a plane-stress matrix cannot create out-of-plane or thickness modes.

Verify tensor symmetry, positive elastic energy, SI units, rigid-body modes and uniform-strain behavior. Benchmark homogeneous propagation against the appropriate anisotropic bulk/rod limits in their regimes. Keep the single-crystal tensor fixed in laboratory coordinates when the beam bends. Geometry rotation and crystal rotation are distinct physical changes.

### 3. Enumerate all channels and measure power

Classify extensional, in-plane and out-of-plane flexural, torsional, mixed and higher transverse/thickness modes using displacement/energy projections and mode/subspace tracking. At each frequency identify every propagating port channel; increase eigenpair-search coverage until the count is stable. Converge evanescent content independently. There is no universal maximum mode count, especially across a tenfold thickness change.

For velocity `v = -i omega u`, the mean elastic energy flux is

$$
J_j=-\tfrac12\operatorname{Re}(\sigma_{ij}v_i^*).
$$

Integrate over the physical port area and check the sign using a known traveling wave. For an orthogonal unit-power channel basis, each incident channel `n` has transmitted power `T_n = sum_m |t_mn|^2` and reflected power `R_n = sum_m |r_mn|^2`. Use the appropriate flux/biorthogonal form when the modal formulation requires it. Account for physical absorption and every other open radiation channel before evaluating `1 - T - R - A`. Numerical sponge absorption is a termination diagnostic, not intrinsic germanium loss.

The calibration gate is a homogeneous lossless matched guide with near-unit transmission, small reflection, complete power accounting and consistent reciprocal channel pairs. Vary mesh, lead length, monitor position and boundary treatment independently. Report both channel conversion and total power. A displacement probe may supplement this analysis but cannot replace it.

### 4. Build controlled frequency, temperature and geometry sweeps

Start the primary propagation interval at **0.1-6 GHz** with **121 frequency points per spectrum**, explicitly inserting cutoffs and relevant feature neighborhoods. Adaptively refine resonances, rapid phase changes and spectral interpolation error. The initial cap of **1001 points** is a planning limit; unresolved spectra remain unresolved. Resolve an isolated linewidth with at least 10-20 samples across it and demonstrate further-refinement stability. Extend to 8 GHz only after validity and resource checks.

The initial straight-device matrix contains 3 thicknesses x 7 proposed temperatures x 2 patterned-device types = **42 planned spectra**, or **5082 initial frequency configurations**, subject to the material admission policy. Homogeneous controls, incident channels, uncertainty cases and convergence repeats are additional. After a deliberate null test, mechanically identical fixed-property cases may reuse one spectrum through explicit operator hashes while applying separate occupation weights.

Only introduce lifetime, attenuation or thermal-transport predictions after the relevant physical model is sourced. A lossless coherent calculation can establish reflection and conversion, not intrinsic phonon lifetime. A transmission integral over 0.1-6 GHz gives a band-limited thermal contribution unless the excluded spectrum is bounded. It is not automatically bulk thermal conductivity.

### 5. Test curvature and circular patterns fairly

First connect an open curved beam to matched straight leads. Screen `R/w = 5, 10, 20, 40` at the three thicknesses and one supported reference temperature, alongside matched straight controls. Parameterize arc length, hole placement, radius, ligament, taper and supports; reject overlaps or infeasible inner-edge clearances. Match path length and local dimensions as closely as possible and disclose any remaining differences.

This adds **12 planned bent-device spectra** before controls. Carry informative cases, including a non-improving case, through the temperature study. Study closed rings and bus-coupled rings later as distinct topologies. Closed-ring eigenmodes do not supply an open device's radiative quality factor. Seek changes in polarization mixing, reflection, confinement, mode splitting and through/drop response that exceed numerical and material uncertainty.

### 6. Establish results that can withstand review

Use at least three systematic meshes for representative cases in every geometry/thickness family. Initial targets are less than 0.5% change in tracked eigenfrequency/gap edge, nondegenerate overlap above 0.99, localization change below 0.02 absolute, and power error plus independently refined `T/R` changes below 0.01 absolute. These are proposed gates, not achieved claims. Tighten them when a scientific comparison requires it. Converge any linewidth, `Q`, attenuation length and mode cutoff separately; establish a numerical floor before reporting deep suppression in dB.

Deliver dispersion and cutoff maps, mode inventories, calibrated scattering matrices, temperature/thickness maps, curvature comparisons, selected displacement/strain/flux fields, uncertainty estimates and resource scaling. Each figure must answer a scientific question and trace to a configuration, source revision and raw table. Include failed and null cases. Publication suitability depends on evidence, convergence and independent scientific review; the number of plots does not establish it.

## Next steps for the continuing AI

1. Read [AGENTS.md](AGENTS.md), [AI handoff](docs/AI_HANDOFF_AP_pnc.md), [evidence/recovery status](docs/EVIDENCE_AND_RECOVERY_AP_pnc.md) and the continuation PDF.
2. Run the available checks; recover missing original files by exact hash before claiming archive reproduction.
3. Complete a sourced material registry and a small, verified 3D homogeneous-lead implementation.
4. Establish mode completeness, power normalization and boundary convergence before patterned-device sweeps.
5. Execute the [ordered work packages](docs/NEXT_STEPS_AP_pnc.md), pilot resource use, retain checkpoints and generate a claim-by-claim evidence ledger.
6. Prepare the detailed final research report using [the report contract](reports/REPORT_REQUIREMENTS_AP_pnc.md), then send it to **athul.prem@coyotes.usd.edu** using an authenticated email service. Record a delivery receipt only after successful sending; otherwise leave a complete delivery-ready package and report delivery as pending.

## Authorship, filenames and new figures

Use **Athul Prem** and **`_AP_pnc`** in new script/document headers and structured metadata. Every project-specific file basename ends `_AP_pnc` before its extension; tool-recognized names such as `README.md`, `AGENTS.md`, `requirements.txt`, `CITATION.cff`, `.gitignore` and `.gitattributes` retain their standard names and contain attribution. Archived bytes stay unchanged and receive per-file attribution in the provenance index.

Every newly exported figure must visibly show `Athul Prem | athul_simulation_pnc_<identifier>` at the top and use a filename such as `athul_simulation_pnc_HPGe_t100nm_T4K_flux_AP_pnc.png`. Preserve historical figures; a relabeled/recomputed image is a new derivative with its own provenance. See [authorship and naming](docs/AUTHORSHIP_AND_NAMING_AP_pnc.md).
