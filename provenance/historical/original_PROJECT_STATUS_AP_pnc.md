# Project status

This file is updated only when a stage has evidence in the project outputs.

| Stage | Status | Evidence |
|---|---|---|
| Environment | PARTIAL | Direct FEM completed with local Python containing Gmsh 4.15.0, NumPy, SciPy, and Matplotlib; no internet installs used |
| Excel validation | PASS | `results/tables/hole_validation.csv`; 26 rows, exact sheet/columns, spacing and gap checks |
| Geometry sanity plots | PASS | `results/figures/pnc_geometry.png`, `pnc_geometry_center_zoom.png` |
| Geometry CAD | PASS | `geometry/ge_pnc_2d.brep`, `.geo_unrolled`; 26 exact circular holes; Gmsh 4.15.0 |
| Mesh | PASS | `mesh/ge_pnc_2d.msh`; 16,386 nodes, 31,340 triangles, required 1500 nm leads, zero triangles inside holes |
| Propagation solver | PASS | Direct NumPy/SciPy plane-stress FEM; 4 GHz residual 1.47e-13 |
| Reference control | PASS | Same direct formulation on `mesh/reference_ge_beam.msh` |
| Frequency sweep | PASS | Five-frequency and 111-point 1–12 GHz sweep completed |
| Periodic/mirror comparison | PASS | Unpatterned, periodic 550 nm mirror, and exact tapered defect; five and full sweeps |
| 2-D Bloch bands | PASS (reduced order) | 550 nm unit cell; one computed complete gap at 2.830–2.936 GHz; no complete 4–8 GHz gap in 16 branches |
| Cavity eigenmode | PASS (reduced order) | 2.923 GHz mode lies in computed reduced-order gap; localization factor 0.252 inside |x|<2.5 um |
| Parameter sweeps | NOT RUN | Deferred until the 2-D band/mode and mesh results are converged; no values fabricated |
| Fabrication tolerance | NOT RUN | Deferred with parameter sweeps; no shifts fabricated |
| Mesh convergence | PARTIAL / NOT CONVERGED | 40/30/20 nm checks completed at 5.5 GHz; 30-to-20 nm P3 change remains ~15.9% |
| 3-D model | NOT RUN | Correctly deferred: 2-D mesh is not converged and no validated 4–8 GHz complete Bloch gap was found |
| Tests | PASS | `scripts/test_direct_fem.py`: 3 internal unittest cases passed |

## Direct FEM result summary

The 4 GHz PnC solve passed with residual `1.4714e-13`. The five-frequency
PnC/reference comparison and the full 111-point 1–12 GHz sweep passed with
finite solutions and residuals below `1e-7`. The minimum full-sweep relative
transmission was `-45.0776 dB` at `3.7 GHz`. These are finite-structure
transmission suppressions, not Bloch bandgaps.

The periodic-mirror finite comparison reached `-28.39 dB` at 5.2 GHz and the
exact tapered defect reached `-33.65 dB` at 5.2 GHz when normalized to the
unpatterned beam. The unit-cell calculation is a reduced-order 2-D Bloch
calculation with 16 lowest branches; its only complete computed gap is
2.830–2.936 GHz. The selected exact-defect eigenmode is 2.923 GHz with
localization factor 0.252 and effective 2-D mode area 6.40 um^2. Its maximum
computed strain is near x=+3.94 um, not at the geometric center.

## Current mesh evidence

The CAD was constructed in nanometer coordinate units to avoid OpenCASCADE
absolute-tolerance loss at SI-scale coordinates; conversion to SI is explicit
in `config.py` and the solver contract. The current lead-inclusive mesh has
16,386 nodes and 31,340 triangles. `results/figures/pnc_mesh.png` is the mesh
visualization. meshio/scikit-fem are not used by the completed direct solver.

## Assumptions

Beam width 1000 nm and 1500 nm leads are provisional. Material parameters are isotropic Ge approximations. The first solver is plane stress and in-plane only. The source has normalized amplitude `F0=1`; it is not a single-phonon amplitude.
