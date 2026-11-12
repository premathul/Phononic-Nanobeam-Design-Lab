# HPGe mode-resolved propagation: scientific specification

Author: **Athul Prem** | File/project tag: **`_AP_pnc`**
Status: **Proposed research; no new 3D results are supplied by this specification.**

## Scientific questions and discriminating tests

| Question | Controlled comparison | Decisive observable |
| --- | --- | --- |
| Does thickness open additional propagation channels or destroy an in-plane stop band? | Same planar pattern/material, 100/200/1000 nm solids | Complete lead cutoffs, 3D dispersion, channel transmission and polarization |
| Is a thermal change mechanical or only statistical? | Fixed-operator null, material-only update, then contraction/prestress | Eigenfrequency and normalized scattering change versus Bose-weighted power |
| Does curvature help confinement or merely increase reflection? | Matched straight/open-bend devices, equal thickness and path length | Total T/R, polarization conversion, localization and uncertainty |
| Does a ring support a useful mode family? | Closed ring, then bus-coupled ring with controlled coupling | Mode identity/splitting, through/drop response and independently converged linewidth |
| Are differences robust to dimensions and material uncertainty? | One-variable screening, followed by selected interactions | Sensitivity derivatives, uncertainty intervals and resolved design ranking |

A null shift, absent complete gap or inferior circular pattern is a permissible scientific outcome. A result needs the comparison that excludes the most plausible numerical alternative, rather than a visually impressive field plot alone.

## Material, coordinate and temperature contract

The proposed material is crystalline Ge with no intentional doping or added layers. Record chemical purity, isotope composition, carrier concentration/resistivity, dislocations/defects and specimen provenance separately. Where absent, use `null` and label the case nominal pure HPGe. Do not turn an unknown property into zero.

Use SI internally. Specify a right-handed laboratory frame, a right-handed crystal frame and an orthogonal rotation matrix with determinant +1. Cubic stiffness has three independent constants. The stability checks in the unstressed cubic basis include `C44 > 0`, `C11 - C12 > 0` and `C11 + 2 C12 > 0`. Verify the implemented tensor symmetries and positive strain energy after rotation. State the Voigt/shear convention so factors of two are unambiguous.

Every property record must identify value/model, units, valid temperature and frequency ranges, measurement method/specimen, citation/page/table, uncertainty and interpolation/extrapolation policy. Density and thermal dilation must conserve mass consistently. An uncertain density or stiffness cannot be hidden by plotting only normalized frequencies.

The nominal historical constants `C11 = 128.5 GPa`, `C12 = 48.3 GPa`, `C44 = 66.8 GPa`, `rho = 5323 kg/m^3` are recorded in the uploaded smoke configuration. They are inputs from the earlier project, not newly established 4 K or millikelvin material measurements. The supplied material template records them only as a fixed-property control. See [primary literature and reading limits](REFERENCES_AP_pnc.md).

The proposed temperature points are 0.02, 0.04, 1, 4, 20, 77 and 300 K. Assign each case exactly one policy:

| Policy | Mechanical input | Allowed interpretation |
| --- | --- | --- |
| `fixed_property_control` | Same tensor, density, geometry and physical damping for all labels | Statistical occupation changes only; mechanics is a null test |
| `material_only_fixed_geometry` | Sourced temperature-dependent properties, fixed geometry | Constitutive contribution to mechanical shifts |
| `free_thermal_contraction` | Supported property/dilation law with a stated reference temperature | Coupled material and free-shape contribution |
| `constrained_thermoelastic` | Equilibrium thermal stress and justified prestressed linearization | Support-constrained contribution, only after additional validation |

For free isotropic expansion, a scale factor may be expressed as `a(T)/a(Tref) = exp(integral alpha_linear(T') dT')` when alpha is the differential linear-expansion coefficient. Do not confuse linear and volumetric alpha; enforce `rho(T) * volume(T)` consistency for fixed mass. A frozen reference geometry with variable density is a distinct sensitivity/constitutive scenario and must say so.

## Governing 3D model and boundary conditions

Start with small-amplitude, linear anisotropic elasticity and the `exp(-i omega t)` convention. With symmetric strain and constitutive tensor `C`, solve `-div(sigma) = rho omega^2 u` in each physical solid. Specify traction-free surfaces, support constraints and port tractions/displacements explicitly. Do not impose both independent traction and displacement on the same boundary without a justified mixed formulation.

Finite-element matrices must be dimensionally consistent and recover rigid translations/rotations appropriate to the boundary conditions. Validate consistent mass, stiffness symmetry, patch behavior and independent small-problem eigenvalues before launching a large sparse eigensolve. Treat prestress and frequency-dependent complex properties as separately verified extensions. For complex stiffness, test time-convention sign, positive cycle-averaged dissipation and attenuation on a simple benchmark.

For a periodic straight device use Bloch boundary conditions with documented cell length and phase convention. Track every branch needed to establish the claimed gap throughout the Brillouin zone. A gap confined to a symmetry sector or the in-plane displacement subspace must be labeled accordingly. Resolve wavevector extrema independently of mesh resolution; a gap inferred from sparse samples can disappear between samples.

## Geometry design and equality of comparisons

The historical workbook supplies 26 hole centers/diameters, not a complete fabricated-device metrology record. The beam width 1000 nm and end leads 1500 nm are provisional historical assumptions. Keep them fixed during the first controlled comparison, then examine uncertainty and width/pitch/taper sensitivity separately. Preserve the source workbook and derive a unit-checked parameter table from it.

Extrude physical thicknesses 100, 200 and 1000 nm with correct free surfaces. Use the same top-view pattern for thickness comparison. A 1000 nm-thick, 1000 nm-wide section is not a thin plate. Include actual fillets, etch profile or roughness only when specified, or as labeled sensitivity models.

For every solid verify closed volumes, finite positive dimensions, no hole overlap, minimum ligaments, correct physical groups and port normals. Refine the exact intended geometry rather than changing polygonal hole shape incidentally with the mesh. Generate at least two independent representative meshes.

For bends, define centerline `s = R theta`, width, thickness, arc angle, hole sequence, local pitch and lead-transition geometry. A given `R/w` can be infeasible for a fixed path length; reject or explicitly redesign it rather than forcing the full matrix. A ring needs a closure condition relating circumference, pitch and hole count; report which quantity changes. State whether a taper creates a defect, because a uniformly periodic ring and a defect ring are different devices.

Use a global crystal tensor for a single-crystal bend. A tensor rotating with the local tangent corresponds to a different physical material model and may only be shown as a labeled diagnostic. For a large-radius open-bend limit, compare against the corresponding straight geometry while accounting for port transitions and path length.

## Lead modes and scattering

Solve the matched lead cross-section eigenproblem at each requested frequency or a controlled dispersion formulation. Identify propagation direction from signed power/energy transport; handle cutoff channels separately. Frequency proximity alone is insufficient for mode identity near crossings. Use displacement/energy projections, symmetry when available, and overlap of invariant subspaces at degeneracy.

Classify extensional, both flexural, torsional, mixed and higher transverse/thickness branches. Do not assume only four modes remain open over the full interval. Increase search coverage until all propagating channels are recovered, and retain enough evanescent content for matching to converge. Archive the search parameters and the point where channel count stabilizes.

Normalize each incident mode by the physical power integral of `J_j = -0.5 Re(sum_i sigma_ij v_i*)`, `v = -i omega u`. Confirm signs with an analytically known traveling wave. In a flux-orthogonal basis, sum squared scattering amplitudes over all outgoing propagating channels; in a nonorthogonal/non-Hermitian formulation use its proper flux bilinear form. State normalization units and polarization conventions.

Check energy conservation including modeled physical absorption and all extra radiation ports. The residual `1 - T - R - A` is meaningful only when the budget is complete. A matched, lossless homogeneous guide should transmit unit incident power within tolerance. Test reciprocal channel pairs in a passive reciprocal model, without assuming arbitrary mode-number labels pair correctly. Vary lead length, monitor position, absorber dimensions/profile, evanescent truncation and spatial mesh independently.

## Frequency sampling, linewidths and floors

Primary interval: 0.1-6 GHz. The 121-point starting grid is a survey, not a completeness proof. Add lead cutoffs, known 2.8/5.2 GHz diagnostic neighborhoods and candidate features explicitly. Refine amplitude, phase and interpolation error. Use the proposed 1001-point cap as a budget checkpoint; store unresolved intervals if it is reached.

For isolated resonances, place at least 10-20 points across an estimated linewidth and refine again. Validate fitted uncertainty, background choice, spectral window and mode identity. Frequency sampling alone may not resolve a very narrow resonance economically; use a separately validated pole/eigenvalue method where appropriate. Do not report `Q = f0 / delta_f` for an unresolved or non-isolated line.

Use `10 log10(T)` for power transmission and `20 log10(amplitude ratio)` for amplitudes only with the appropriate equal-impedance context. The historical ratio is an amplitude-probe diagnostic. Report absolute errors near deep minima and establish a validated floor with stronger controls; do not claim attenuation below that floor.

## Temperature, dissipation and conditional thermal transport

Compute `n(f,T) = 1 / expm1(h f / (k_B T))` with stable large/small-argument handling. A frozen mechanical operator must give the same normalized scattering matrix at different temperature labels. Deliberately solve two points as a null test, then reuse exactly matching mechanical operators with an explicit provenance link and separate statistical weights.

Classical coherent reflection/conversion and incoherent transport are different regimes. Compare device length against justified mean-free-path estimates before treating high-temperature transport as ballistic. Intrinsic loss needs a supported anharmonic/isotope/defect/surface model and validity interval; a numerical sponge is not that model. Extract propagation attenuation using matched devices of multiple lengths to distinguish entrance reflection from bulk decay. State amplitude versus power attenuation and fitting intervals; avoid noise-floor points.

If ballistic or otherwise justified transmission is available, the band-limited linear-response conductance is

$$
G_{\rm band}(T)=\int_{f_{\min}}^{f_{\max}} h f\,\mathcal{T}(f)
\frac{\partial n(f,T)}{\partial T}\,df,\qquad
\mathcal{T}(f)=\operatorname{Tr}(t^\dagger t).
$$

Here `f` is ordinary frequency in Hz; do not mix it with angular frequency in the integration measure. This formulation is motivated by ballistic phonon transport [Rego and Kirczenow, 1998](https://doi.org/10.1103/PhysRevLett.81.232). A limited GHz interval cannot be called total conductance at all temperatures without a spectral-tail bound. Conductance is not bulk thermal conductivity.

## Validation and uncertainty contract

| Quantity/gate | Initial target | Qualification |
| --- | --- | --- |
| Small-problem dimensionless algebraic residual | Approximately 1e-8 or better | Record conditioning and attained accuracy |
| Spatial convergence | At least 3 systematic meshes | Separate hole curvature, wavelength and thickness resolution |
| Tracked frequency/gap edge | <0.5% final change with stable trend | Refine wavevector sampling independently |
| Nondegenerate overlap | >0.99 | Use subspace overlap at degeneracy |
| Localization fraction | <0.02 absolute change | State the integration volume and energy normalization |
| Power-budget error | <0.01 incident power | Count all physical outgoing channels |
| T/R sensitivity | <0.01 absolute under each independent refinement | Boundary, mesh and modal truncation separately |
| Linewidth, Q, attenuation | Independently converged and uncertainty estimated | No generic frequency gate certifies them |
| Design comparison | Difference exceeds combined uncertainty | Tighten targets or report unresolved ranking |

Numerical uncertainty, material uncertainty and dimensional uncertainty need separate entries. A tolerance is not an error bar by itself; show observed refinement trends and sensitivity. Use independent solver/operator comparisons on small identical problems where practical. Experimental validation requires an external experimental benchmark; absent one, describe results as numerically verified within the stated model.

The final evidence ledger must retain rejected points, solver failures, omitted channels, unsupported temperatures and unresolved spectral intervals. A failed point is not zero transmission and must not be filled by interpolation in the numerical table.

## Dataset contract

Each case stores author/tag, run ID, parent/operator hashes, source commit, exact inputs, resolved units, geometry parameters, thickness, temperature/policy, crystal frame, material provenance, mesh/port settings, solver tolerances, resource limits, start/finish status and failure diagnostics. Each frequency/channel row stores incident/outgoing identities, cutoff/propagation status, power normalization, T/R/absorption/budget, polarization, wavenumber/velocity, residual, DOFs and convergence flags.

Use explicit values for `planned`, `running`, `completed_unverified`, `verified_within_model`, `failed` and `blocked`; reserve experimental validation for its own evidence field. Absent physical channels, failed computations and uncomputed points are different states. Save completed tasks atomically and resume only matching input/source identities. Full fields are required for representative passband, gap, resonance, conversion and failure cases, not for every point by default.

Required tables and figure families are listed in [the ordered task plan](NEXT_STEPS_AP_pnc.md) and [final report contract](../reports/REPORT_REQUIREMENTS_AP_pnc.md). Every new plot carries Athul Prem and its `athul_simulation_pnc_` identity at the top.
