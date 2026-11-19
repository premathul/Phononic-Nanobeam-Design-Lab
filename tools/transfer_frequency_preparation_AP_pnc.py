#!/usr/bin/env python3
"""Transfer a verified Windows native preparation to Linux. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools import seed_frequency_map_AP_pnc as seeder

native = seeder.native
workflow = native.workflow
METHOD = "windows_to_linux_native_preparation"


def _publication(donor, state):
    """Bind the independently published map and its report/table to this checkpoint."""
    published = donor / native.MAP
    publication_sha256 = native._sha(published)
    result = native.read_saved_map(donor)
    checkpoint_sha256 = native._sha(donor / native.CHECKPOINT)
    if (result["checkpoint_sha256"] != checkpoint_sha256
            or any(result[key] != state[key] for key in ("inputs", "inputs_sha256", "shared", "source_git_commit"))
            or len(result["points"]) != len(state["points"])
            or any(any(saved.get(key) != value for key, value in point.items())
                   for saved, point in zip(result["points"], state["points"], strict=True))
            or native._sha(published) != publication_sha256):
        raise ValueError("Donor publication differs from its native checkpoint.")
    return checkpoint_sha256, publication_sha256


def export_preparation(donor: Path, receipt: Path) -> str:
    donor, receipt = Path(donor).resolve(strict=True), Path(receipt).resolve()
    if receipt.is_relative_to(donor):
        raise ValueError("Save the transfer receipt outside the original donor directory.")
    source, _, provenance = seeder._verify_donor(donor)
    identity = source["inputs"]["numerical_identity"]
    if not identity["platform"].startswith("Windows"):
        raise ValueError("Export requires a verified native Windows donor on its original host environment.")

    state = native.read_map_checkpoint(donor)
    checkpoint_sha256, publication_sha256 = _publication(donor, state)
    if checkpoint_sha256 != provenance["donor_checkpoint_sha256"]:
        raise ValueError("The donor checkpoint changed during export.")
    record = {"schema_version": 1, **provenance,
              "exporter_source_sha256": native._sha(Path(__file__).resolve()),
              "donor_publication_sha256": publication_sha256,
              "donor_numerical_identity": identity,
              "donor_thread_environment": source["inputs"]["thread_environment"]}
    contents = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with receipt.open("xb") as stream:
        stream.write(contents)
    return hashlib.sha256(contents).hexdigest()


def _receiver_identity(origin):
    implementation = workflow.source_identity()
    identity = native._numerical_identity(implementation)
    identity["source_sha256"] = {**identity["source_sha256"], **{
        path.relative_to(ROOT).as_posix(): native._sha(path)
        for path in (Path(native.__file__).resolve(), Path(native.comparison.__file__).resolve())}}
    if not origin["platform"].startswith("Windows") or not identity["platform"].startswith("Linux"):
        raise ValueError("Import requires a Windows donor and a Linux execution environment.")
    if origin["source_sha256"] != identity["source_sha256"]:
        raise ValueError("Linux numerical source bytes differ from the original Windows preparation.")
    if origin["packages"] != identity["packages"]:
        raise ValueError("Linux numerical package versions differ from the original Windows preparation.")
    if origin["python"].split(".")[:2] != identity["python"].split(".")[:2]:
        raise ValueError("Linux Python major/minor version differs from the original Windows preparation.")
    return implementation["git_commit"], identity


def _check_section(directory, shared, section_path, config, length_m):
    mesh, manifest = workflow.load_checkpoint(directory, shared["mesh_hash"])
    order = config["mesh"]["element_order"]
    element = native.ElementVector(native.ElementTetP2() if order == 2 else native.ElementTetP1())
    layout = native._DisplacementDofLayout(mesh, element, native.Dofs(mesh, element), 2 * order)
    if layout.N != manifest["ndof"] or layout.N > config["resources"]["max_dofs"]:
        raise ValueError("The restored mesh displacement layout differs or exceeds the configured body cap.")
    triangles = mesh.facets[:, mesh.boundaries["xleft"]]
    vertices = np.unique(triangles)
    section = native.MeshTri(mesh.p[1:, vertices], np.searchsorted(vertices, triangles))
    element = native.ElementVector(native.ElementTriP2() if order == 2 else native.ElementTriP1(), dim=3)
    basis = native.Basis(section, element, intorder=2 * order)
    if basis.N > config["resources"]["max_port_dofs"]:
        raise ValueError("The restored section exceeds the configured port cap.")
    tolerance = max(length_m, float(np.ptp(mesh.p, axis=1).max())) * (64 * np.finfo(float).eps)
    with np.load(section_path, allow_pickle=False) as arrays:
        if not np.array_equal(arrays["points_m"], section.p) or not np.array_equal(arrays["triangles"], section.t):
            raise ValueError("The restored section vertex or triangle order differs from the native left trace.")
        for side, position in (("left", 0.), ("right", length_m)):
            indices = arrays[f"{side}_dofs"]
            boundary = mesh.boundaries[f"x{side}"]
            expected_dofs = layout.get_dofs(facets=boundary).all()
            if (indices.shape != (basis.N,) or indices.dtype.kind not in "iu"
                    or not np.array_equal(np.sort(indices), np.sort(expected_dofs))
                    or basis.N != manifest["end_face_dofs"][f"x{side}"]
                    or not np.allclose(layout.doflocs[0, indices], position, atol=tolerance, rtol=0)
                    or not np.allclose(layout.doflocs[1:, indices], basis.doflocs, atol=tolerance, rtol=0)
                    or not all(np.isin(indices[local], body).all()
                               for local, body in zip(basis.split_indices(), layout.split_indices(), strict=True))):
                raise ValueError(f"The restored {side} section displacement order differs from the native mesh trace.")


def _verified_transfer(donor, receipt, receipt_sha256):
    contents = receipt.read_bytes()
    if hashlib.sha256(contents).hexdigest() != receipt_sha256:
        raise ValueError("Transfer receipt SHA-256 differs from the pinned export receipt.")
    record = json.loads(contents)
    if (record["schema_version"] != 1
            or record["exporter_source_sha256"] != native._sha(Path(__file__).resolve())
            or record["seeder_source_sha256"] != native._sha(Path(seeder.__file__).resolve())):
        raise ValueError("Transfer receipt schema or source identity differs.")
    state = native.read_map_checkpoint(donor)
    checkpoint_sha256, publication_sha256 = _publication(donor, state)
    inputs, shared = state["inputs"], state["shared"]
    origin = inputs["numerical_identity"]
    if (record["donor_checkpoint_sha256"] != checkpoint_sha256
            or record["donor_publication_sha256"] != publication_sha256
            or record["donor_shared_sha256"] != workflow.digest(shared)
            or record["donor_source_git_commit"] != state["source_git_commit"]
            or record["donor_numerical_identity"] != origin
            or record["donor_thread_environment"] != inputs["thread_environment"]):
        raise ValueError("Transferred donor checkpoint, publication or preparation differs from the receipt.")
    revision, identity = _receiver_identity(origin)
    config = workflow.resolve_config(inputs["configuration"])
    geometry, holes, pattern = workflow._resolve_geometry(config)
    if workflow.digest(config) != workflow.digest(inputs["configuration"]) or pattern != inputs["hole_pattern"]:
        raise ValueError("Transferred configuration or original hole source differs on Linux.")
    directory, mesh_hash, mesh_inputs = workflow._mesh_location(Path(), geometry, pattern, config["mesh"], origin)
    artifacts = shared["artifacts"]
    stage = Path(artifacts["mass"]["path"]).parent
    filenames = (directory / native.RUN_SUMMARY, directory / "mesh_AP_pnc.npz", Path("K.npz"), Path("M.npz"),
                 Path("port_section_AP_pnc.npz"))
    if (shared["mesh_hash"] != mesh_hash or shared["reference_planes_x_m"] != [0., geometry.length_m]
            or artifacts.keys() != set(native.SHARED_FILES) or len(stage.parts) != 1
            or not stage.name.startswith("shared-")
            or any(artifacts[name]["path"] != (stage / filename).as_posix()
                   for name, filename in zip(native.SHARED_FILES, filenames, strict=True))):
        raise ValueError("Transferred native preparation layout, mesh identity or reference planes differ.")
    paths = {name: native._verified(artifacts[name], donor) for name in native.SHARED_FILES}
    manifest = native._read(paths["mesh_summary"])
    if (workflow.digest(manifest["mesh_inputs"]) != workflow.digest(mesh_inputs)
            or manifest["mesh_sha256"] != artifacts["mesh"]["sha256"]
            or manifest["hole_pattern_provenance"] != pattern or manifest["hole_count"] != len(holes)
            or paths["mesh"] != paths["mesh_summary"].parent / "mesh_AP_pnc.npz"):
        raise ValueError("Transferred mesh does not bind the original Windows inputs and hole geometry.")
    _check_section(paths["mesh"].parent, shared, paths["port_section"], config, geometry.length_m)
    provenance = {"method": METHOD, "receipt_sha256": receipt_sha256, "receipt": record}
    return inputs, shared, revision, identity, paths, provenance


def import_preparation(donor: Path, receipt: Path, receipt_sha256: str, output: Path, frequencies_ghz) -> dict:
    """Install the exact Windows files in a new native Linux map with no attempted points."""
    donor, output = Path(donor).resolve(strict=True), Path(output).resolve()
    if output.is_relative_to(donor) or donor.is_relative_to(output):
        raise ValueError("Select separate donor and Linux map directories.")
    frequencies = list(frequencies_ghz)
    if (not frequencies or any(not workflow._finite_number(f) or f <= 0 or not math.isfinite(f * 1e9) for f in frequencies)
            or any(b * 1e9 <= a * 1e9 for a, b in zip(frequencies, frequencies[1:]))):
        raise ValueError("Supply strictly increasing positive frequency samples in GHz.")
    initialize = not output.exists() or (output.is_dir() and not any(output.iterdir()))
    if not initialize and not (output / native.CHECKPOINT).is_file():
        raise ValueError("Select an empty output or the same Linux transfer map.")
    original, shared, revision, identity, paths, provenance = _verified_transfer(donor, Path(receipt), receipt_sha256)
    inputs = {**original, "frequencies_GHz": frequencies, "numerical_identity": identity,
              "thread_environment": {key: os.getenv(key) for key in native.THREADS}}
    state = {**native.AUTHOR, **native.CONVERGENCE, "schema_version": 1, "inputs_sha256": workflow.digest(inputs),
             "inputs": inputs, "source_git_commit": revision, "shared": shared, "preparation_reuse": provenance,
             "points": [{"index": i, "frequency_GHz": f, "status": "pending", "attempts": [], **native.CONVERGENCE}
                        for i, f in enumerate(frequencies, 1)]}
    if initialize:
        seeder._install_preparation(output, state, paths)
    if any(path.resolve() != path for path in output.iterdir()):
        raise ValueError("The output contains an entry linked outside its native path.")
    saved = native.read_map_checkpoint(output)
    if any(saved.get(key) != state[key] for key in ("inputs_sha256", "shared", "preparation_reuse")):
        raise ValueError("Existing Linux map differs from the verified donor or requested frequencies.")
    return native.map_scattering(inputs["configuration"], frequencies, output, max_new_points=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Verify a Windows donor and write a new receipt.")
    receive = commands.add_parser("import", help="Install verified preparation in a separate Linux map.")
    for command in (export, receive):
        command.add_argument("--from-map", type=Path, required=True, dest="donor")
        command.add_argument("--receipt", type=Path, required=True)
    receive.add_argument("--receipt-sha256", required=True)
    receive.add_argument("--output", type=Path, required=True)
    receive.add_argument("--frequencies-ghz", type=float, nargs="+", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            checksum = export_preparation(args.donor, args.receipt)
            print(f"Receipt SHA-256: {checksum}")
        else:
            import_preparation(args.donor, args.receipt, args.receipt_sha256, args.output, args.frequencies_ghz)
            print(f"Verified Linux native preparation: {args.output.resolve()}")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"Could not transfer frequency preparation: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
