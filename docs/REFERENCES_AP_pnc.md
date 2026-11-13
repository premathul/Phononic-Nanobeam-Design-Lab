# Primary references and source-use boundaries

Author: **Athul Prem** | Project/file tag: **`_AP_pnc`**

These references are a starting reading list for the proposed material/transport registry. Bibliographic records and available primary abstracts were checked during preparation. No numerical temperature curve or loss model was extracted and installed. Obtain the full papers and relevant corrections before entering parameter tables. The existence of a citation does not certify the proposed GHz/millikelvin/nanoscale regime.

| Source | Relevance | Boundary for use |
| --- | --- | --- |
| M. E. Fine, *Elastic Constants of Germanium between 1.7 and 80 K*, J. Appl. Phys. 26, 862-863 (1955), [10.1063/1.1722110](https://doi.org/10.1063/1.1722110) | Candidate low-temperature cubic elastic-property source | Obtain primary full text and applicable correction before extracting constants. The title's lower limit is 1.7 K; it is not direct millikelvin data. |
| B. I. Miller, *Temperature Dependence of the Ultrasonic Attenuation in Germanium*, Phys. Rev. 132, 2477 (1963), [10.1103/PhysRev.132.2477](https://doi.org/10.1103/PhysRev.132.2477) | Publisher abstract describes longitudinal [111] attenuation measured at 4-132 MHz, 20-300 K | Useful for loss mechanisms and validity checks; not direct calibration for GHz nanoscale propagation at mK temperatures. Full parameter extraction still required. |
| W. P. Mason and T. B. Bateman, *Ultrasonic-Wave Propagation in Pure Silicon and Germanium*, JASA 36, 644-652 (1964), [10.1121/1.1919031](https://doi.org/10.1121/1.1919031) | Publisher abstract distinguishes pure specimens below 1e14 impurity atoms/cm^3 and examines phonon-interaction attenuation | Keep specimen purity, mode and experimental regime in the registry; do not transplant a bulk ultrasonic fit without justification. |
| L. G. C. Rego and G. Kirczenow, *Quantized Thermal Conductance of Dielectric Quantum Wires*, PRL 81, 232 (1998), [10.1103/PhysRevLett.81.232](https://doi.org/10.1103/PhysRevLett.81.232) | Primary Landauer ballistic-phonon transport framework | Supports a conditional ballistic conductance calculation; it does not establish that every planned temperature/device is ballistic. |
| R. Colella, B. Reinhart, E. Alp and E. E. Haller, *Thermal Expansion of Germanium Isotopes at Low Temperature*, [arXiv:0910.1569v2](https://arxiv.org/abs/0910.1569v2) | Primary experimental preprint on isotope-specific low-temperature expansion | Distinguish natural and monoisotopic Ge. Read the exact sample/temperature ranges before adopting an expansion law. |

### Further material-source task

Investigate H. J. McSkimin, J. Appl. Phys. 24, 988-997 (1953), [10.1063/1.1721449](https://doi.org/10.1063/1.1721449), as a candidate broader-temperature elastic source. Obtain its primary text and distinguish the ranges for different materials; do not read a fused-silica temperature range as a Ge measurement range. Fine's correction and any later precision measurements must be explicitly checked in WP1.

### Internal sources

- [Supplied continuation brief](references/Phononic_Research_Continuation_AP_pnc.pdf): scientific scope and proposed gates; no execution evidence for the new campaign.
- [Historical results summary](../archive/documentation/RESULTS_AND_NEXT_GOALS_AP_pnc.md): model lineage and previously reported successes/failures.
- [Available smoke configuration](../results/archive_smoke/run_new_verified_smoke/config_AP_pnc.json): nominal parameters and explicit calibration limitations.
- [Available evidence audit](EVIDENCE_AND_RECOVERY_AP_pnc.md): distinguishes actually rechecked outputs from historical reports.

For every numerical property eventually used, record the source page/figure/table, exact extracted values, extraction uncertainty, original measurement units and any conversion. Preserve a provenance trail to an authorized source copy or stable publisher link. Literature values, numerical verification and experimental validation are separate evidence categories.

## Wave accuracy and mesh controls

The device follow-up uses the sources below to guide targeted body and port checks. They motivate separate phase and energy checks; their mesh-size recommendations are not calibration data for this anisotropic Ge device.

| Source | Relevance | Boundary for use |
| --- | --- | --- |
| D. L. Brown and D. Gallistl, *Multiscale Sub-grid Correction Method for Time-Harmonic High-Frequency Elastodynamics with Wavenumber Explicit Bounds*, [arXiv:1608.04243v2](https://arxiv.org/abs/1608.04243v2) | Documents frequency-dependent pollution error in elastodynamics and a multiscale treatment | Supports treating propagation-phase accuracy as a separate requirement. No multiscale method or transferable element-size threshold from this paper has been installed here. |
| S. Geevers, W. A. Mulder and J. J. W. van der Vegt, *Dispersion Properties of Explicit Finite Element Methods for Wave Propagation Modelling on Tetrahedral Meshes*, J. Sci. Comput. 77, 372-396 (2018), [10.1007/s10915-018-0709-7](https://doi.org/10.1007/s10915-018-0709-7) | Examines elastic-wave dispersion, resolution and tetrahedron-shape sensitivity | Its mass-lumped and discontinuous-Galerkin time-domain schemes differ from this consistent-mass P2 harmonic solver. The lesson is to measure dispersion and element quality, not copy a points-per-wavelength rule. |
| [Gmsh mesh-size fields](https://gmsh.info/doc/texinfo/gmsh.html#t10_003a-Mesh-size-fields) and [structured meshes](https://gmsh.info/doc/texinfo/gmsh.html#t6_003a-Transfinite-meshes) | Established controls for volume targets and prescribed surface subdivisions | The existing mesher holds port and circle subdivisions separately from the body target. Holding the number of wall layers fixed does not by itself fix the complete volume mesh. |

In the current study, a matched-port wavenumber difference multiplied by a stated guide length is a propagation-phase sensitivity indicator. It is not a bound on the patterned device's scattering error, and it is not subtracted from the measured complex scattering difference. Direct saved-result comparisons retain the phase at the original physical reference planes.
