"""Join saved periodic-cell operators at one frequency. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import json
from math import tau
from numbers import Integral
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

if __name__ == "__main__":
    os.environ.update(dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"), "1"))

import numpy as np
from scipy.linalg import solve
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from phononic.lead_modes_AP_pnc import _match_transverse_points
from phononic.port_modes_AP_pnc import PortModes, PortSystem, assemble_port, solve_port_modes
from phononic.resumable_scattering_AP_pnc import (
    AUTHOR, THREADS, _array_digest, _exclusive, _link, _numerical_identity, _read, _sha, _verified,
)
from phononic.scattering_AP_pnc import solve_scattering
from phononic.workflow_AP_pnc import digest, measured_resources, source_identity

RESIDUAL_LIMIT = 1e-8
POWER_LIMIT = 0.01
NATIVE_COMPARISON_LIMIT = 1e-8


def _complex_values(values):
    return {"real": values.real.tolist(), "imag": values.imag.tolist()}


def _relative_error(residual, scale):
    denominator = np.max(np.abs(scale), axis=-2)
    numerator = np.max(np.abs(residual), axis=-2)
    return float(np.max(numerator / np.where(denominator > 0, denominator, 1.0)))


def _flux(displacements, outward_forces, omega):
    work = displacements.conj().T @ outward_forces
    return (omega / (4j)) * (work - work.conj().T)


def _native_trace_order(arrays, port):
    coordinates = arrays["trace_dof_yz_m"]
    components = arrays["trace_dof_components"]
    size = port.basis.N
    if (coordinates.shape != (size, 2) or not np.isfinite(coordinates).all()
            or components.shape != (size,) or not np.isin(components, [0, 1, 2]).all()):
        raise ValueError("Cell and native lead transverse traces differ.")
    native_coordinates = port.basis.doflocs
    span = np.ptp(native_coordinates, axis=1)
    order = np.empty(size, dtype=int)
    for component, native in enumerate(port.basis.split_indices()):
        saved = np.flatnonzero(components == component)
        order[native] = saved[_match_transverse_points(coordinates[saved].T, native_coordinates[:, native], span)]
    selected = np.ix_(order, order)
    trace_mass = arrays["trace_mass"]
    if (trace_mass.shape != (size, size)
            or not np.allclose(trace_mass[selected], port.M, rtol=0, atol=1e-12 * np.max(abs(port.M)))):
        raise ValueError("Cell and native lead trace mass or triangles differ.")
    return selected


def solve_finite_mirror_response(
    cell_artifact: Path | str, port: PortSystem | PortModes, cell_count: int,
) -> dict:
    """Solve every native left incident mode on a chain of one to eight cells.

    LL/LR/RL/RR in the saved cell are exact discrete Schur blocks in N/m.
    Neighboring cells share a full interface displacement and their outward
    reactions sum to zero there. Both end maps include the full outgoing SAFE
    basis, with all evanescent columns, at the unrephased end planes.
    """
    if isinstance(cell_count, bool) or not isinstance(cell_count, Integral) or not 1 <= cell_count <= 8:
        raise ValueError("Finite mirror response requires one to eight identical cells.")
    cell_count = int(cell_count)
    native_port = port.port if isinstance(port, PortModes) else port
    with np.load(cell_artifact, allow_pickle=False) as arrays:
        frequency = float(arrays["frequency_hz"].item())
        period = float(arrays["period_m"].item())
        if not np.isfinite(frequency) or frequency <= 0 or not np.isfinite(period) or period <= 0:
            raise ValueError("The cell frequency and period must be positive and finite.")
        selected = _native_trace_order(arrays, native_port)
        size = int(native_port.basis.N)
        cell_blocks = [arrays[name] for name in ("LL", "LR", "RL", "RR")]
        if any(block.shape != (size, size) or not np.isrealobj(block)
               or not np.isfinite(block).all() for block in cell_blocks):
            raise ValueError("Finite mirror response requires four finite real dynamic boundary blocks.")
        ll, lr, rl, rr = (block[selected] for block in cell_blocks)
    cell_dynamic = np.block([[ll, lr], [rl, rr]])
    symmetry_error = _relative_error(cell_dynamic - cell_dynamic.T, cell_dynamic)
    if symmetry_error >= RESIDUAL_LIMIT:
        raise ValueError("The saved lossless cell failed dynamic-block symmetry.")

    modes = port if isinstance(port, PortModes) else solve_port_modes(port, frequency)
    if modes.frequency_hz != frequency:
        raise ValueError("Native lead and condensed mirror frequency differ.")
    omega = tau * frequency
    left_out, right_out = modes.negative_indices, modes.positive_indices
    reflection_ids = left_out[modes.propagating[left_out]]
    incident_ids = right_out[modes.propagating[right_out]]
    if not incident_ids.size:
        raise ValueError("The native lead has no propagating left incident modes.")
    displacements, axial_forces = modes.displacements, modes.axial_forces
    u_left, u_right = displacements[:, left_out], displacements[:, right_out]
    f_left, f_right = -axial_forces[:, left_out], axial_forces[:, right_out]
    incident_u, incident_f = displacements[:, incident_ids], -axial_forces[:, incident_ids]
    incoming_power = -_flux(incident_u, incident_f, omega)
    powers = incoming_power.diagonal().real
    if not np.all(np.isfinite(powers) & (powers > 0)):
        raise ValueError("The native incident fields must carry inward power.")

    z_left = solve(u_left.T, f_left.T).T
    z_right = solve(u_right.T, f_right.T).T
    blocks = [[None] * (cell_count + 1) for _ in range(cell_count + 1)]
    blocks[0][0], blocks[-1][-1] = csr_matrix(ll - z_left), csr_matrix(rr - z_right)
    seam_diagonal = csr_matrix(rr + ll)
    for seam in range(1, cell_count):
        blocks[seam][seam] = seam_diagonal
    upper, lower = csr_matrix(lr), csr_matrix(rl)
    for cell in range(cell_count):
        blocks[cell][cell + 1] = upper
        blocks[cell + 1][cell] = lower
    operator = bmat(blocks, format="csc")
    rhs = np.zeros((operator.shape[0], incident_ids.size), dtype=complex)
    rhs[:size] = incident_f - z_left @ incident_u
    fields = splu(operator).solve(rhs)
    if not np.isfinite(fields).all():
        raise RuntimeError("Finite mirror solve produced a nonfinite displacement.")
    trace = fields.reshape(cell_count + 1, size, incident_ids.size)
    # Recover original single-cell reactions before any DtN or seam blocks are added.
    left_reactions = ll @ trace[:-1] + lr @ trace[1:]
    right_reactions = rl @ trace[:-1] + rr @ trace[1:]
    left_force, right_force = left_reactions[0], right_reactions[-1]
    reflected_u = trace[0] - incident_u
    reflected_f = left_force - incident_f
    reflect_all, transmit_all = solve(u_left, reflected_u), solve(u_right, trace[-1])
    reflect = reflect_all[modes.propagating[left_out]]
    transmit = transmit_all[modes.propagating[right_out]]

    physical_r = _flux(reflected_u, reflected_f, omega)
    physical_t = _flux(trace[-1], right_force, omega)
    outward = _flux(trace[0], left_force, omega) + physical_t
    modal_r = -reflect_all.conj().T @ modes.flux_gram(left_out) @ reflect_all
    modal_t = transmit_all.conj().T @ modes.flux_gram(right_out) @ transmit_all
    r_channel = abs(reflect)**2 * abs(modes.powers_w[reflection_ids, None]) / powers
    t_channel = abs(transmit)**2 * abs(modes.powers_w[incident_ids, None]) / powers
    reflect_power, transmit_power = r_channel.sum(axis=0), t_channel.sum(axis=0)
    power_scale = np.sqrt(powers[:, None] * powers)
    coherence_defect = (physical_r + physical_t - incoming_power) / power_scale
    operator_scale = float(np.asarray(abs(operator).sum(axis=1)).max())
    residuals = {
        "dynamic_block_symmetry": symmetry_error,
        "native_safe_eigenpair": float(np.max(modes.relative_residuals)),
        "joined_equation": _relative_error(operator @ fields - rhs, operator_scale * abs(fields) + abs(rhs)),
        "interface_force_balance": 0.0,
        "end_displacement_reconstruction": max(
            _relative_error(u_left @ reflect_all - reflected_u, abs(trace[0]) + abs(incident_u)),
            _relative_error(u_right @ transmit_all - trace[-1], trace[-1]),
        ),
        "end_traction_reconstruction": max(
            _relative_error(f_left @ reflect_all - reflected_f, abs(left_force) + abs(incident_f)),
            _relative_error(f_right @ transmit_all - right_force, right_force),
        ),
    }
    if cell_count > 1:
        seam_left, seam_right = left_reactions[1:], right_reactions[:-1]
        residuals["interface_force_balance"] = _relative_error(
            seam_left + seam_right, abs(seam_left) + abs(seam_right),
        )
    power_errors = {
        "modal_energy_budget": float(np.max(abs(1.0 - reflect_power - transmit_power))),
        "coherent_energy_budget": float(np.linalg.norm(coherence_defect, ord=2)),
        "coherent_net_end_flux": float(np.linalg.norm(outward / power_scale, ord=2)),
        "coherent_modal_traction_flux_agreement": float(max(
            np.linalg.norm((physical_r - modal_r) / power_scale, ord=2),
            np.linalg.norm((physical_t - modal_t) / power_scale, ord=2),
        )),
        "propagating_channel_traction_flux_agreement": float(max(
            np.max(abs(physical_r.diagonal().real / powers - reflect_power)),
            np.max(abs(physical_t.diagonal().real / powers - transmit_power)),
        )),
    }
    accepted = max(residuals.values()) < RESIDUAL_LIMIT and max(power_errors.values()) < POWER_LIMIT
    return {
        "status": "accepted" if accepted else "unresolved", "frequency_hz": frequency, "period_m": period,
        "cell_count": cell_count, "reference_planes_m": [0.0, cell_count * period],
        "native_port": {"trace_dof_count": size, "outgoing_columns_per_end": len(left_out),
                        "evanescent_columns_per_end": len(left_out) - len(reflection_ids),
                        "operator_sha256": _array_digest(native_port.K0, native_port.G, native_port.Q, native_port.M),
                        "trace_sha256": _array_digest(native_port.basis.doflocs, native_port.left_dofs, native_port.right_dofs),
                        "outgoing_displacement_condition_numbers": [float(np.linalg.cond(u_left)), float(np.linalg.cond(u_right))]},
        "native_incident_root_ids": incident_ids.tolist(), "native_reflection_root_ids": reflection_ids.tolist(),
        "native_transmission_root_ids": incident_ids.tolist(), "incident_powers_w": powers.tolist(),
        "reflection_amplitudes": _complex_values(reflect), "transmission_amplitudes": _complex_values(transmit),
        "reflection": reflect_power.tolist(), "transmission": transmit_power.tolist(),
        "direct_reflection_flux": (physical_r.diagonal().real / powers).tolist(),
        "direct_transmission_flux": (physical_t.diagonal().real / powers).tolist(),
        "coherent_power_defect": _complex_values(coherence_defect),
        "relative_residuals": residuals, "power_errors": power_errors,
        "relative_residual_limit": RESIDUAL_LIMIT, "power_error_limit": POWER_LIMIT,
    }


def compare_native_response(result, native):
    """Compare coefficients using matching native SAFE bases and physical planes."""
    if (result["frequency_hz"] != native.frequency_hz or native.incident_side != "left"
            or result["native_incident_root_ids"] != native.incident_mode_indices.tolist()
            or result["native_reflection_root_ids"] != native.reflection_mode_indices.tolist()
            or result["native_transmission_root_ids"] != native.transmission_mode_indices.tolist()):
        raise ValueError("Uncondensed native response uses different frequency, side or root IDs.")
    differences = {}
    for key in ("reflection_amplitudes", "transmission_amplitudes", "reflection", "transmission"):
        saved = result[key]
        if isinstance(saved, dict):
            saved = np.asarray(saved["real"]) + 1j * np.asarray(saved["imag"])
        differences[key] = float(np.max(abs(saved - getattr(native, key))))
    native_residual = float(np.max(native.relative_residuals))
    passed = max(differences.values()) < NATIVE_COMPARISON_LIMIT and native_residual < RESIDUAL_LIMIT
    return {"status": "accepted" if passed else "unresolved", "maximum_absolute_changes": differences,
            "native_original_equation_residual": native_residual, "absolute_limit": NATIVE_COMPARISON_LIMIT}


def run_saved_case(cell_directory: Path, periods: int, output: Path, *, native_check: bool = False):
    """Verify a saved cell, solve one chain and publish one immutable receipt."""
    from tools import run_mirror_cell_phase_AP_pnc as cell

    if native_check and periods != 1:
        raise ValueError("The native full-volume countercheck uses one cell.")
    folder = cell_directory.resolve()
    output = output.resolve()
    receipt_path = folder / cell.RECEIPT
    saved = _read(receipt_path)
    if saved["status"] != "accepted" or saved["inputs_sha256"] != digest(saved["inputs"]):
        raise ValueError("The source cell must have an accepted receipt with unchanged inputs.")
    archive = _verified(saved["trace_arrays"], folder)
    if output in (receipt_path, archive):
        raise ValueError("The finite mirror receipt must have its own path.")
    with np.load(archive, allow_pickle=False) as arrays:
        if (arrays["inputs_sha256"].item() != saved["inputs_sha256"]
                or arrays["frequency_hz"].item() != saved["inputs"]["frequency_hz"]):
            raise ValueError("Condensed blocks and saved cell receipt have different inputs.")
    source = source_identity()
    for path in (Path(__file__).resolve(), Path(cell.__file__).resolve()):
        source["source_sha256"][path.relative_to(ROOT).as_posix()] = _sha(path)
    inputs = {
        "cell_receipt": _link(receipt_path, folder), "cell_trace_arrays": saved["trace_arrays"],
        "cell_inputs_sha256": saved["inputs_sha256"], "periods": periods, "native_check": native_check,
        "numerical_identity": _numerical_identity(source), "threads": THREADS,
    }
    identity = digest(inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive(output.with_name(f".{output.name}.lock")):
        if output.exists():
            previous = _read(output)
            if previous["inputs_sha256"] != identity or digest(previous["inputs"]) != identity:
                raise ValueError("Existing finite mirror receipt uses different inputs.")
            if previous["result_sha256"] != digest(previous["result"]):
                raise ValueError("Existing finite mirror result changed.")
            return {**previous, "reused": True}

        layout = saved["inputs"]["mesh"]
        geometry = saved["inputs"]["geometry"]
        controls = (geometry["thickness_nm"], layout["central_target_nm"], layout["thickness_layers"])
        spec = cell._case_spec(*controls)
        if spec.inputs != {key: saved["inputs"][key] for key in spec.inputs}:
            raise ValueError("Current cell geometry, material or mesh controls differ from the saved cell.")
        with measured_resources() as resources:
            system = cell.assemble_case(*controls)
            if _array_digest(system.mesh.p, system.mesh.t) != saved["mesh"]["geometry_connectivity_sha256"]:
                raise ValueError("Rebuilt native port uses a different finite-cell mesh.")
            native_port = assemble_port(system, spec.material, max_dofs=layout["max_face_dofs"])
            modes = solve_port_modes(native_port, spec.inputs["frequency_hz"])
            result = solve_finite_mirror_response(archive, modes, periods)
            if native_check:
                direct = solve_scattering(system, modes, modes.frequency_hz)
                result["native_countercheck"] = compare_native_response(result, direct)
                if result["native_countercheck"]["status"] != "accepted":
                    result["status"] = "unresolved"
        record = {**AUTHOR, "schema_version": 1, "inputs": inputs, "inputs_sha256": identity,
                  "cell_directory": str(folder), "source_git_commit": source["git_commit"],
                  "cell_geometry": geometry, "cell_mesh_controls": layout, "resources": resources,
                  "result": result, "result_sha256": digest(result),
                  "scope": "Finite repeated outer-mirror segment with uniform leads at its original end planes."}
        with NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=output.parent, delete=False) as stream:
            stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
        Path(stream.name).replace(output)
        return {**record, "reused": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-directory", required=True, type=Path)
    parser.add_argument("--periods", required=True, type=int, choices=range(1, 9))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native-check", action="store_true", help="Compare one cell against its full-volume solution.")
    args = parser.parse_args()
    saved = run_saved_case(args.cell_directory, args.periods, args.output, native_check=args.native_check)
    print(json.dumps({"status": saved["result"]["status"], "output": str(args.output.resolve()),
                      "reused": saved["reused"]}, allow_nan=False))
    return 0 if saved["result"]["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
