"""Solve one complementary regional mesh control. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from tempfile import mkdtemp

if __name__ == "__main__":
    os.environ.update(dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                     "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"))

import numpy as np
from scipy.sparse import save_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import central_patch_refinement_AP_pnc as patch
from tools import map_scattering_AP_pnc as native
from tools import run_fixed_wall_response_AP_pnc as single_point
from tools.compare_scattering_AP_pnc import require

workflow = native.workflow


def _prepare(output, config, pattern, identity, mesh, diagnostics, source, region):
    geometry, holes, original = workflow._resolve_geometry(config)
    require(original == pattern and len(holes) == 26, "Donor pattern differs from the historical device.")
    control = {"donors": source, "refined_region": region, "mesh_geometry_connectivity_sha256":
               diagnostics["control_coordinates_and_connectivity_sha256"]}
    pattern = {**pattern, "taper_patch_refinement": control}
    mesh_inputs = {**workflow.mesh_inputs(geometry, pattern, config["mesh"], identity),
                   "taper_patch_refinement": control}
    mesh_hash = workflow.digest(mesh_inputs)
    stage = Path(mkdtemp(prefix="shared-", dir=output))
    directory = stage / "_meshes" / mesh_hash
    manifest = {**native.AUTHOR, "status": "mesh_ready", "mesh_hash": mesh_hash, "mesh_inputs": mesh_inputs,
                "hole_pattern_provenance": pattern, "hole_count": len(holes),
                "geometry_diagnostics": {"taper_patch_refinement": diagnostics},
                **{key: diagnostics[key] for key in single_point.MESH_COUNTS}}
    mesh, manifest, _ = workflow.save_checkpoint(directory, mesh, manifest)
    params = config["material"]
    material = workflow.CubicMaterial(C11=params["C11_Pa"], C12=params["C12_Pa"], C44=params["C44_Pa"],
                                      rho=params["density_kg_m3"], rotation=params["crystal_rotation"])
    with workflow.measured_resources() as assembly_resources:
        system = workflow._assemble_system(mesh, config, material)
    with workflow.measured_resources() as section_resources:
        port = workflow.assemble_port(system, material, max_dofs=config["resources"]["max_port_dofs"])
    require(system.K.shape == system.M.shape == (diagnostics["ndof"], diagnostics["ndof"]),
            "Hybrid displacement space differs from mesh preflight.")
    paths = [directory / native.RUN_SUMMARY, directory / "mesh_AP_pnc.npz",
             stage / "K.npz", stage / "M.npz", stage / "port_section_AP_pnc.npz"]
    save_npz(paths[2], system.K, compressed=False)
    save_npz(paths[3], system.M, compressed=False)
    np.savez(paths[4], points_m=port.basis.mesh.p, triangles=port.basis.mesh.t, left_dofs=port.left_dofs,
             right_dofs=port.right_dofs, K0=port.K0, M=port.M, Q=port.Q, G=port.G,
             reference_speed_m_s=port.reference_speed_m_s)
    record = {"mesh_hash": mesh_hash, "assembly_resources": assembly_resources, "section_resources": section_resources,
              "reference_planes_x_m": [0.0, geometry.length_m],
              "artifacts": {key: native._link(p, output) for key, p in zip(native.SHARED_FILES, paths, strict=True)}}
    return native.SharedInputs((mesh, manifest), directory, port, record), pattern


def run(coarse, fine, output, *, prepare_only=False, region="central"):
    coarse, fine, output = (Path(p).resolve() for p in (coarse, fine, output))
    require(all(not output.is_relative_to(donor) and not donor.is_relative_to(output) for donor in (coarse, fine)),
            "Choose an output directory separate from both native donors.")
    require(region in ("central", "complement"), "Choose the central patch or its exact complement.")
    threads = {name: os.environ.get(name) for name in native.THREADS}
    require(threads == native.THREADS, "Use one numerical thread.")
    revision = workflow.source_identity()
    identity = native._numerical_identity(revision)
    identity["source_sha256"] = dict(identity["source_sha256"])
    for path in (Path(__file__), Path(patch.__file__), Path(native.__file__), Path(single_point.__file__)):
        identity["source_sha256"][path.relative_to(ROOT).as_posix()] = native._sha(path)
    coarse_state = native.read_map_checkpoint(coarse)
    fine_state = native.read_map_checkpoint(fine)
    coarse_mesh, fine_mesh, source = patch.load_nested_patch_sources(coarse, fine)
    config = coarse_state["inputs"]["configuration"]
    candidate = fine_state["inputs"]["configuration"]
    require(all(config[key] == candidate[key] for key in ("geometry", "material", "analysis"))
            and config["geometry"]["thickness_nm"] in (100, 200) and config["mesh"]["thickness_layers"] == 2
            and tuple(s["mesh"]["central_max_edge_nm"] for s in (config, candidate)) in
                ((225, 168.75), (168.75, 126.5625))
            and all(config["mesh"][k] == candidate["mesh"][k] for k in config["mesh"] if k != "central_max_edge_nm"),
            "Choose adjacent saved two-layer device meshes of one thickness from the 225/168.75/126.5625 nm hierarchy.")
    pattern = coarse_state["inputs"]["hole_pattern"]
    require(pattern == fine_state["inputs"]["hole_pattern"], "Donor hole patterns differ.")
    output.mkdir(parents=True, exist_ok=True)
    with native._exclusive(output / ".frequency_map.lock"):
        checkpoint = output / native.CHECKPOINT
        if checkpoint.exists():
            state = native.read_map_checkpoint(output)
            state.pop("checkpoint_data_sha256")
            require(state["taper_patch_donors"] == source and state.get("refined_region", "central") == region
                    and state["inputs"]["numerical_identity"] == identity
                    and state["inputs"]["thread_environment"] == threads, "Existing patch inputs or numerical source differ.")
            if state["points"][0]["status"] == "accepted":
                native._read_accepted(state["points"][0], state["inputs"]["configuration"], state["shared"], output)
                if (output / native.MAP).exists():
                    published = native.read_saved_map(output)
                    if published["checkpoint_sha256"] == native._sha(checkpoint):
                        return published
            shared = native._restore_shared(state["shared"], state["inputs"]["configuration"], output)
        else:
            require(all(p.name == ".frequency_map.lock" or (p.name.startswith("shared-") and p.is_dir())
                        for p in output.iterdir()), "Output contains a different study.")
            mesh, checks = patch.make_central_patch_control(coarse_mesh, fine_mesh,
                                                          tuple(source["nominal_mirror_cuts_x_m"]), region=region)
            device = "thin100" if config["geometry"]["thickness_nm"] == 100 else "thick200"
            patch_name = "taper" if region == "central" else "complement"
            name = f"{device}_{patch_name}_patch_AP_pnc"
            config = workflow.resolve_config({**config, "name": name,
                                              "resources": {**config["resources"], "max_dofs": checks["ndof"]}})
            shared, pattern = _prepare(output, config, pattern, identity, mesh, checks, source, region)
            inputs = {"configuration": config, "hole_pattern": pattern, "frequencies_GHz": [1.0],
                      "numerical_identity": identity, "thread_environment": threads}
            state = {**native.AUTHOR, **native.CONVERGENCE, "schema_version": 1, "inputs": inputs,
                     "inputs_sha256": workflow.digest(inputs), "source_git_commit": revision["git_commit"],
                     "taper_patch_donors": source, "refined_region": region, "shared": shared.record,
                     "points": [{"index": 1, "frequency_GHz": 1.0, "status": "pending", "attempts": [], **native.CONVERGENCE}]}
            native._write_checkpoint(checkpoint, state)
        if prepare_only and state["points"][0]["status"] == "pending":
            return native._publish(state, shared, {}, output)
        accepted = single_point._run_or_recover(state, output, shared, identity)
        return native._publish(state, shared, accepted, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", required=True, type=Path)
    parser.add_argument("--fine", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--region", choices=("central", "complement"), default="central")
    args = parser.parse_args()
    saved = run(args.coarse, args.fine, args.output, prepare_only=args.prepare_only, region=args.region)
    point, = saved["points"]
    print(f"{point['status']}: {args.output.resolve()}")
    if point["status"] == "failed":
        print(point["attempts"][-1])
    raise SystemExit(0 if point["status"] == "accepted" or (args.prepare_only and point["status"] == "pending") else 1)
