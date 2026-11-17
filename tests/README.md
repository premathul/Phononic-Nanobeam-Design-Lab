# Active tests

Author: **Athul Prem** | Project/file tag: **`_AP_pnc`**

Run `python -m unittest discover -s tests -p 'test*_AP_pnc.py' -v` from the repository root. [test_nanobeam_AP_pnc.py](test_nanobeam_AP_pnc.py) preserves four elementary geometry/analytic-chain checks. [test_archive_runtime_AP_pnc.py](test_archive_runtime_AP_pnc.py) checks path/hash/alias/overwrite/completeness behavior. These tests verify the stated utility and evidence-preservation behavior; they do not validate the proposed 3D research campaign.

[test_scattering_AP_pnc.py](test_scattering_AP_pnc.py) checks the complete two-port, power-normalized scattering matrix of a small uniform guide: `S.conj().T @ S = I`. This tests cross-channel orthogonality as well as the separate incident-channel power budgets. An unequal complex superposition incident from both ends also checks outgoing power from the finite-domain nodal forces, independently of the outgoing modal amplitudes. The test preserves the original port reference planes and uses the same port basis for both solves. It verifies lossless coherent-drive consistency for this bounded case, not spatial convergence, reciprocity, or the full patterned device.

The Lawrence h60 workflow tests use small systems to verify exact reconstruction of staged equations, recovery without solving accepted drives again, and rejection of altered data or invalid residuals. The saved-run comparison tests check the reviewed Linux phase and solver provenance. Full-device mesh convergence requires its own physical comparison. [Staging and recovery tests](test_lawrence_device_refinement_AP_pnc.py) · [Saved-run comparison tests](test_compare_scattering_AP_pnc.py)


[Thickness-control tests](test_lawrence_thickness_AP_pnc.py) verify exact in-plane footprint preservation, a SAFE section built without body matrices, variable incident-root identities, interrupted accepted-prefix recovery, and refusal of altered staged equations. The new z4/z6 comparator profile keeps source/SAFE provenance and physical mode matching separate from its scientific acceptance result.
