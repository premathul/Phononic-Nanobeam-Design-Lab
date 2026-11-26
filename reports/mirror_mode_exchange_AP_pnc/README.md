# Mode-selective reflection in the periodic Ge nanobeam at 1 GHz

Athul Prem | _AP_pnc | 26 November 2026

## Result

In this lossless numerical model of a 100 nm thick periodic Ge mirror, the dominant one-to-eight-cell increase in reflection occurs between the third and fifth propagating lead branches, ranked by |k| in each direction. The converted channel carries 0.8476% at one period and 37.2102% at eight periods for either pure unit-power launch. The two same-rank reflections at eight periods are 1.1599% and 0.0040%. This extends the finite-mirror transmission comparison by identifying where the reflected power goes.

A separate Bloch calculation using the saved cell operators identifies a decaying pair in the same transverse symmetry class as the selected lead branches (even under the full vector width reflection and odd under thickness reflection). The two Bloch traces have 99.8357–99.8403% mass-weighted squared projection onto the *four-dimensional displacement-trace span* of the two signed lead branches. This projection is not a power fraction. On five archived cell discretizations, the pair has |λ| = 0.91940–0.93370 and phases ±2.248 to ±2.251 radians per period. The three resolved propagating Bloch pairs at |λ| = 1 belong to the other transverse symmetry classes; each has less than 0.0022% of its trace norm squared in the selected span. The condensed polynomial residual for the compared roots is below 2.2e−13. The attenuation per period remains mesh dependent.

For the 255-DOF lead, k3 = 3.9608444 × 10^6 and k5 = 7.1392039 × 10^6 rad/m, giving (k3 + k5)a − 2π ≈ −0.17816 rad for a = 550 nm. This phase proximity is consistent with intermodal coupling, but does not by itself demonstrate a Bragg mechanism. On the fixed mesh, the power loss is reflection, not dissipative loss. The numerical finding is a symmetry-selective response at one frequency, not a complete bandgap, a bandwidth measurement, quantitative mesh convergence, or a tapered-device result.

## Figure caption

(a) Fraction of unit incident power reflected into selected backward branches, using the same archived cell and port for lengths 1 through 8. The exchanged paths 3→5 and 5→3 overlap at plotting precision. (b) Decaying Bloch pair in the selected symmetry class and the three propagating pairs of other symmetry classes, with degenerate moduli at each setting plotted once. The top two rows are independent Mac and Windows mesh realizations at the same edge/layer settings. The other rows are archived refinement settings. Geometry: 550 nm period, 250 nm through-hole diameter, 1 µm width and 100 nm thickness, nominal cubic Ge with identity crystal rotation. Neither panel represents experimental data.

## Reproduce

Use source and data at commit `240c6805ead982890b84e76b4dd6aa2c21f5674e` of [Phononic-Nanobeam-Design-Lab](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/tree/240c6805ead982890b84e76b4dd6aa2c21f5674e). The archived trace digest is checked in each run. Version receipts are in results/*.json. Python requires numpy, scipy, matplotlib, scikit-fem, gmsh, meshio and its dependencies. From the repo root, with PKG an absolute path to this package:

```sh
export PYTHONPATH=".:src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p /tmp/mirror_pair_fresh
for case in reports/mirror_length_AP_pnc results/runs/mirror_cell_phase_AP_pnc/t100_z*; do
  python "$PKG/mode_pair_AP_pnc.py" "$case" "/tmp/mirror_pair_fresh/${case##*/}.json"
done
python "$PKG/plot_AP_pnc.py" .
```

The Schur blocks solve [RL + λ(LL + RR) + λ²LR]u = 0; the script obtains the propagation factors by homogeneous generalized eigendecomposition. It assembles the native uniform lead, matches its transverse points and trace mass to each frozen archive, then obtains signed wavenumbers and displacement branches. It estimates reflection symmetry by reflecting the coordinates and the vector component normal to each symmetry plane, using finite-element interpolation. Projection means x†M Q(Q†M Q)^+Q†M x / x†M x, where Q contains the four signed traces of ranks 3 and 5, x is the Bloch trace, and M is the cross-sectional mass matrix.

The separate endpoint replay uses the Mac cell blocks with a cloud-rebuilt lead. Do not claim identical floating-point arrays or root labels across platforms: the matched trace mass differs by 6.83e−17 of its largest entry, and the largest difference in the one- and eight-cell reflected or transmitted coefficient powers is 3.20e−13 of incident power. The native propagating branches were compared in wavenumber order. No source solver or archived record was modified.
