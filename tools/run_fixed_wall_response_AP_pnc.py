#!/usr/bin/env python3
"""Prepare and run the fixed-wall 1 GHz device control. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import mkdtemp

import numpy as np
from scipy.sparse import save_npz

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools import map_scattering_AP_pnc as native
from tools import transfer_frequency_preparation_AP_pnc as transfer
from tools.compare_scattering_AP_pnc import require

workflow = native.workflow
RECEIPT = "fixed_wall_preparation_AP_pnc.json"
WALL_SOURCE = ROOT / "tools/fixed_wall_faceting_AP_pnc.py"
METHOD = "fixed_24_sided_wall_with_48_mesh_edges"
FREQUENCY_GHZ = 1.0
CHANNELS = 5
DONOR_CHECKPOINT_SHA256 = {
    100: "094c4a0abd0c45589b7ce3e3a139adaf2a423a0d8023ca12cc8c1f3a148876d6",
    200: "e8598e99ba049ef5051fcb77022f57e93f26c05ab7d8a95ec30868f413f7970e",
}
# These donor-era files predate central-edge refinement. The coefficient kernels and
# native point runner still match, and the archived geometry is verified independently.
DONOR_LEGACY_SOURCES = {f"src/phononic/{name}_AP_pnc.py" for name in ("layered", "mesh_checkpoint", "workflow")}
SECTION_ARRAYS = ("points_m", "triangles", "left_dofs", "right_dofs", "K0", "M", "Q", "G", "reference_speed_m_s")
MESH_COUNTS = ("ndof", "mesh_vertices", "mesh_tetrahedra", "end_face_dofs", "boundary_facet_counts")


def _identity():
    implementation = workflow.source_identity()
    identity = native._numerical_identity(implementation)
    identity["source_sha256"] = {**identity["source_sha256"], **{
        path.relative_to(ROOT).as_posix(): native._sha(path)
        for path in (Path(__file__).resolve(), WALL_SOURCE, Path(native.__file__).resolve(),
                     Path(native.comparison.__file__).resolve(), Path(transfer.__file__).resolve())}}
    threads = {name: os.getenv(name) for name in native.THREADS}
    require(threads == native.THREADS, "Set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS and BLIS_NUM_THREADS to 1.")
    return implementation["git_commit"], identity, threads


def _new_state(preparation, preparation_sha256):
    inputs = preparation["inputs"]
    return {**native.AUTHOR, **native.CONVERGENCE, "schema_version": 1,
            "inputs": inputs, "inputs_sha256": workflow.digest(inputs),
            "source_git_commit": preparation["assembly_source_git_commit"], "shared": preparation["shared"],
            "fixed_wall_preparation_sha256": preparation_sha256,
            "points": [{"index": 1, "frequency_GHz": FREQUENCY_GHZ, "status": "pending",
                        "attempts": [], **native.CONVERGENCE}]}


def _assemble(donor, output, identity, threads, revision):
    from tools import fixed_wall_faceting_AP_pnc as wall

    donor_state = native.read_map_checkpoint(donor)
    checkpoint_sha, map_sha = transfer._publication(donor, donor_state)
    original = donor_state["inputs"]
    config = original["configuration"]
    thickness_nm = config["geometry"]["thickness_nm"]
    settings = config["mesh"]
    pattern = original["hole_pattern"]
    donor_shared = donor_state["shared"]
    donor_point, = (point for point in donor_state["points"] if point["frequency_GHz"] == FREQUENCY_GHZ)
    require(checkpoint_sha == DONOR_CHECKPOINT_SHA256.get(thickness_nm) and donor_point["status"] == "accepted"
            and [settings[name] for name in ("kind", "element_order", "mesh_size_nm", "lead_mesh_size_nm",
                                             "circle_resolution", "thickness_layers")] == ["layered", 2, 400, 150, 48, 2]
            and "central_max_edge_nm" not in settings and len(pattern["holes"]) == 26
            and config["analysis"]["incident_side"] == "left"
            and config["resources"].get("linear_solver", "direct") == "direct",
            "Choose one of the two pinned 100 or 200 nm, 400/150 nm, two-layer, 48-edge native donors.")
    _, donor_response = native._read_accepted(donor_point, config, donor_shared, donor)
    require(donor_response["num_incident_channels"] == CHANNELS, "The donor must have five accepted incident channels.")
    donor_identity = original["numerical_identity"]
    require(all(identity["source_sha256"].get(path) == sha for path, sha in donor_identity["source_sha256"].items()
                if path not in DONOR_LEGACY_SOURCES)
            and identity["packages"] == donor_identity["packages"],
            "Numerical source and package versions differ from the accepted donor.")
    donor_section = native._verified(donor_shared["artifacts"]["port_section"], donor)
    donor_manifest = native._read(native._verified(donor_shared["artifacts"]["mesh_summary"], donor))

    with workflow.measured_resources() as mesh_resources:
        mesh, geometry, holes, donor_geometry = wall.load_wall_faceting_source(donor)
        changed, diagnostics = wall.make_fixed_wall_faceting_control(
            mesh, geometry, holes, thickness_layers=settings["thickness_layers"],
            patterned_x_interval_m=tuple(value * 1e-9 for value in settings["patterned_x_interval_nm"]))
    require(donor_geometry["donor_checkpoint_sha256"] == checkpoint_sha
            and not np.array_equal(mesh.p, changed.p) and np.array_equal(mesh.t, changed.t)
            and changed.boundaries.keys() == mesh.boundaries.keys()
            and all(np.array_equal(mesh.boundaries[name], changed.boundaries[name]) for name in mesh.boundaries)
            and donor_shared["reference_planes_x_m"] == [0., geometry.length_m],
            "The control must move wall geometry on the original topology and preserve both planes.")
    fixed_wall = {
        "method": METHOD,
        "polygon_sides_per_hole": 24,
        "boundary_edges_per_hole": 48,
        "source_sha256": identity["source_sha256"][WALL_SOURCE.relative_to(ROOT).as_posix()],
        "donor_mesh_sha256": donor_shared["artifacts"]["mesh"]["sha256"],
        "deformed_coordinates_and_connectivity_sha256": native._array_digest(changed.p, changed.t),
        "geometry_diagnostics_sha256": workflow.digest(diagnostics),
    }
    # The configuration's 48 records the unchanged number of boundary edges.
    # The explicit mesh and hole provenance carry the different polygon shape.
    label = "thin" if thickness_nm == 100 else "thick"
    config = workflow.resolve_config({**config, "name": f"{label}{thickness_nm:.0f}_fixed_wall24_on48_AP_pnc"})
    pattern = {**pattern, "fixed_wall_geometry": fixed_wall}
    mesh_inputs = {**workflow.mesh_inputs(geometry, pattern, settings, identity), "fixed_wall_geometry": fixed_wall}
    mesh_hash = workflow.digest(mesh_inputs)
    stage = Path(mkdtemp(prefix="shared-", dir=output))
    directory = stage / "_meshes" / mesh_hash
    manifest = {**native.AUTHOR, "status": "mesh_ready", "mesh_hash": mesh_hash, "mesh_inputs": mesh_inputs,
                "hole_pattern_provenance": pattern, "hole_count": len(holes), "mesh_resources": mesh_resources,
                "geometry_diagnostics": {"fixed_wall_faceting": diagnostics},
                **{key: donor_manifest[key] for key in MESH_COUNTS}}
    changed, manifest, _ = workflow.save_checkpoint(directory, changed, manifest)
    material_settings = config["material"]
    material = workflow.CubicMaterial(C11=material_settings["C11_Pa"], C12=material_settings["C12_Pa"],
                                      C44=material_settings["C44_Pa"], rho=material_settings["density_kg_m3"],
                                      rotation=material_settings["crystal_rotation"])
    with workflow.measured_resources() as assembly_resources:
        system = workflow._assemble_system(changed, config, material)
    with workflow.measured_resources() as section_resources:
        port = workflow.assemble_port(system, material, max_dofs=config["resources"]["max_port_dofs"])
    section = {"points_m": port.basis.mesh.p, "triangles": port.basis.mesh.t,
               "left_dofs": port.left_dofs, "right_dofs": port.right_dofs,
               "K0": port.K0, "M": port.M, "Q": port.Q, "G": port.G,
               "reference_speed_m_s": port.reference_speed_m_s}
    with np.load(donor_section, allow_pickle=False) as baseline:
        require(all(np.array_equal(baseline[name], section[name]) for name in SECTION_ARRAYS)
                and system.K.shape == system.M.shape == (manifest["ndof"], manifest["ndof"]),
                "The fixed-wall geometry changed the donor section matrices, trace order or body displacement layout.")
    stiffness_path, mass_path, section_path = (stage / name for name in ("K.npz", "M.npz", "port_section_AP_pnc.npz"))
    save_npz(stiffness_path, system.K, compressed=False)
    save_npz(mass_path, system.M, compressed=False)
    np.savez(section_path, **section)
    paths = (directory / native.RUN_SUMMARY, directory / "mesh_AP_pnc.npz", stiffness_path, mass_path, section_path)
    shared = {"mesh_hash": mesh_hash, "assembly_resources": assembly_resources,
              "section_resources": section_resources, "reference_planes_x_m": donor_shared["reference_planes_x_m"],
              "artifacts": {key: native._link(path, output) for key, path in zip(native.SHARED_FILES, paths, strict=True)}}
    inputs = {"configuration": config, "frequencies_GHz": [FREQUENCY_GHZ], "hole_pattern": pattern,
              "numerical_identity": identity, "thread_environment": threads}
    return {**native.AUTHOR, "schema_version": 1, "method": METHOD, "inputs": inputs, "shared": shared,
            "assembly_source_git_commit": revision, "original_end_sections_match_exactly": True,
            "donor": {"checkpoint_sha256": checkpoint_sha, "map_sha256": map_sha, "geometry": donor_geometry,
                      "accepted_summary": donor_point["accepted"]["summary"], "mesh_hash": donor_shared["mesh_hash"],
                      "mesh": donor_shared["artifacts"]["mesh"], "port_section": donor_shared["artifacts"]["port_section"],
                      "source_git_commit": donor_state["source_git_commit"]}}


def _read_preparation(output, expected):
    path = output / RECEIPT
    require(path.resolve() == path and native.comparison.is_sha256(expected), "Supply the SHA-256 of the fixed-wall preparation receipt.")
    body = path.read_bytes()
    require(hashlib.sha256(body).hexdigest() == expected, "The fixed-wall preparation receipt differs from the pinned copy.")
    receipt = json.loads(body)
    require(receipt["schema_version"] == 1 and receipt["method"] == METHOD
            and receipt["inputs"]["frequencies_GHz"] == [FREQUENCY_GHZ]
            and receipt["original_end_sections_match_exactly"] is True,
            "Invalid fixed-wall preparation receipt.")
    return receipt


def prepare(donor, output):
    donor, output = Path(donor).resolve(strict=True), Path(output).resolve()
    require(not output.is_relative_to(donor) and not donor.is_relative_to(output),
            "Choose a separate output directory for the fixed-wall control.")
    revision, identity, threads = _identity()
    output.mkdir(parents=True, exist_ok=True)
    with native._exclusive(output / ".frequency_map.lock"):
        receipt_path = output / RECEIPT
        checkpoint = output / native.CHECKPOINT
        if not receipt_path.exists():
            require(all(path.name == ".frequency_map.lock" or (path.name.startswith("shared-") and path.is_dir())
                        for path in output.iterdir()), "Choose a new output folder or this fixed-wall preparation.")
            workflow.write_json(receipt_path, _assemble(donor, output, identity, threads, revision))
        receipt_sha = native._sha(receipt_path)
        preparation = _read_preparation(output, receipt_sha)
        checkpoint_sha, map_sha = transfer._publication(donor, native.read_map_checkpoint(donor))
        require(preparation["donor"]["checkpoint_sha256"] == checkpoint_sha
                and preparation["donor"]["map_sha256"] == map_sha
                and preparation["inputs"]["numerical_identity"] == identity
                and preparation["inputs"]["thread_environment"] == threads,
                "The existing fixed-wall preparation belongs to a different donor or produces different source or environment identities.")
        state = _new_state(preparation, receipt_sha)
        if not checkpoint.exists():
            native._write_checkpoint(checkpoint, state)
        saved = native.read_map_checkpoint(output)
        require(saved["fixed_wall_preparation_sha256"] == receipt_sha and saved["shared"] == state["shared"],
                "The fixed-wall native checkpoint does not match its receipt.")
        for artifact in state["shared"]["artifacts"].values():
            native._verified(artifact, output)
        if not (output / native.MAP).exists():
            require(saved["points"] == state["points"], "Publish a started fixed-wall control with the solve command.")
            native._publish(state, None, {}, output)
        return receipt_sha


def _point_summary(summary_path, point, settings, shared, output, identity, pattern):
    point["accepted"] = {"summary": native._link(summary_path, output)}
    try:
        accepted = native._read_accepted(point, settings, shared, output)
        summary = accepted[1]
        source = summary["implementation"]["source_sha256"]
        require(summary["num_incident_channels"] == CHANNELS
                and summary["hole_pattern_provenance"] == pattern
                and summary["assembly_resources"] == shared["assembly_resources"]
                and all(identity["source_sha256"].get(path) == sha for path, sha in source.items())
                and all(summary["implementation"][key] == identity[key] for key in ("platform", "python", "packages")),
                "The fixed-wall point differs from the expected physical inputs or solve environment.")
    except BaseException:
        point.pop("accepted")
        raise
    return accepted


def _run_or_recover(state, output, shared, identity):
    checkpoint = output / native.CHECKPOINT
    point = state["points"][0]
    settings = state["inputs"]["configuration"]
    pattern = state["inputs"]["hole_pattern"]
    shared_record = shared.record
    if point["status"] == "accepted":
        return {1: native._read_accepted(point, settings, shared_record, output)}
    if point["status"] == "running":
        prior = point["attempts"][-1]
        interrupted = (output / prior["path"]).resolve()
        require(interrupted.is_relative_to(output), "The interrupted attempt is outside its native map.")
        completed = [path for path in interrupted.glob(f"*/{native.RUN_SUMMARY}")
                     if native._read(path)["status"] == "completed_unverified"]
        require(len(completed) <= 1, "More than one completed response exists inside the interrupted attempt.")
        if completed:
            accepted = _point_summary(completed[0], point, settings, shared_record, output, identity, pattern)
            point["status"] = prior["status"] = "accepted"
            native._write_checkpoint(checkpoint, state)
            return {1: accepted}
        prior.update(status="interrupted", error="Process exited before committing the point.")

    point_directory = output / "point-001"
    point_directory.mkdir(exist_ok=True)
    attempt = Path(mkdtemp(prefix="attempt-", dir=point_directory))
    record = {"status": "running", "path": attempt.relative_to(output).as_posix()}
    point["attempts"].append(record)
    point["status"] = "running"
    native._write_checkpoint(checkpoint, state)
    try:
        summary = native._run_point(native._point_config(settings, FREQUENCY_GHZ), shared, attempt)
        summary_path = Path(summary["output_directory"]) / native.RUN_SUMMARY
        accepted = _point_summary(summary_path, point, settings, shared_record, output, identity, pattern)
    except KeyboardInterrupt:
        point["status"] = "interrupted"
        record.update(status="interrupted", error="Interrupted before committing the point.")
        native._write_checkpoint(checkpoint, state)
        native._publish(state, shared, {}, output)
        raise
    except (ArithmeticError, OSError, RuntimeError, ValueError) as exc:
        point["status"] = "failed"
        record.update(status="failed", failure_kind=getattr(exc, "kind", "scattering"), error=str(exc))
        native._write_checkpoint(checkpoint, state)
        return {}
    point["status"] = record["status"] = "accepted"
    native._write_checkpoint(checkpoint, state)
    return {1: accepted}


def solve(output, preparation_sha256):
    output = Path(output).resolve(strict=True)
    checkpoint = output / native.CHECKPOINT
    revision, identity, threads = _identity()
    with native._exclusive(output / ".frequency_map.lock"):
        preparation = _read_preparation(output, preparation_sha256)
        state = native.read_map_checkpoint(output)
        state.pop("checkpoint_data_sha256")
        expected = _new_state(preparation, preparation_sha256)
        origins = preparation["inputs"]
        origin = origins["numerical_identity"]
        require(origin["source_sha256"] == identity["source_sha256"] and origin["packages"] == identity["packages"]
                and origin["python"].split(".")[:2] == identity["python"].split(".")[:2]
                and origins["thread_environment"] == threads
                and (origin == identity or (origin["platform"].startswith("Windows") and identity["platform"].startswith("Linux"))),
                "Solve with the original numerical sources, packages and Python series on the preparation host or a verified Linux receiver.")
        require(all(state[key] == expected[key] for key in ("shared", "fixed_wall_preparation_sha256")),
                "The native map and external preparation receipt differ.")
        settings = origins["configuration"]
        execution_inputs = {**origins, "numerical_identity": identity, "thread_environment": threads}
        execution_sha = workflow.digest(execution_inputs)
        provenance = {"method": METHOD, "receipt_sha256": preparation_sha256,
                      "assembly_numerical_identity": origin,
                      "assembly_source_git_commit": preparation["assembly_source_git_commit"],
                      "original_end_sections_match_exactly": preparation["original_end_sections_match_exactly"]}
        if "preparation_reuse" not in state:
            require(state["inputs_sha256"] == expected["inputs_sha256"] and state["points"] == expected["points"],
                    "A newly solved map must match the original pending preparation.")
            state.update(inputs=execution_inputs, inputs_sha256=execution_sha,
                         source_git_commit=revision, preparation_reuse=provenance)
            native._write_checkpoint(checkpoint, state)
        else:
            require(state["inputs_sha256"] == execution_sha and state["preparation_reuse"] == provenance,
                    "The checkpoint belongs to a different solve environment or preparation.")

        point = state["points"][0]
        if point["status"] == "accepted" and (output / native.MAP).exists():
            published = native.read_saved_map(output)
            if published["checkpoint_sha256"] == native._sha(checkpoint):
                native._read_accepted(point, settings, state["shared"], output)
                return published
        shared = native._restore_shared(state["shared"], settings, output)
        manifest = shared.checkpoint[1]
        pattern = origins["hole_pattern"]
        fixed_wall = pattern["fixed_wall_geometry"]
        require(manifest["hole_pattern_provenance"] == pattern
                and manifest["mesh_inputs"]["fixed_wall_geometry"] == fixed_wall
                and manifest["mesh_inputs"]["settings"] == settings["mesh"]
                and manifest["mesh_inputs"]["pattern"] == {key: pattern[key] for key in ("source_sha256", "holes")}
                and native._array_digest(shared.checkpoint[0].p, shared.checkpoint[0].t)
                == fixed_wall["deformed_coordinates_and_connectivity_sha256"],
                "The restored mesh does not reproduce the declared fixed-wall geometry and source.")
        # The pattern remains in the mesh and accepted summary. Check it separately
        # while leaving the native shared record unchanged for independent replay.
        accepted = _run_or_recover(state, output, shared, identity)
        return native._publish(state, shared, accepted, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preparation = subparsers.add_parser("prepare", help="Assemble the fixed-wall mesh and save the pending native map.")
    preparation.add_argument("--donor", type=Path, required=True, help="Pinned native 100 or 200 nm, 48-edge map.")
    execution = subparsers.add_parser("solve", help="Run or resume one point from a pinned preparation.")
    execution.add_argument("--preparation-sha256", required=True)
    for command in (preparation, execution):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            print(f"{RECEIPT} SHA-256 {prepare(args.donor, args.output)}", flush=True)
            return 0
        result = solve(args.output, args.preparation_sha256)
        point = result["points"][0]
        print(f"{point['status']} at 1 GHz. Report {args.output.resolve() / native.REPORT}", flush=True)
        if point["status"] == "failed":
            print(point["attempts"][-1]["error"], file=sys.stderr)
        return 0 if point["status"] == "accepted" else 2
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, StopIteration) as exc:
        print(f"Fixed-wall control could not complete. {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
