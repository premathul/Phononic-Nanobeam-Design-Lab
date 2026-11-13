"""Internal geometry checkpoints for mesh preflight. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from skfem import MeshTet

from .geometry_AP_pnc import make_straight_mesh


def mesh_inputs(geometry, pattern: dict | None, settings: dict, implementation: dict) -> dict:
    kind = settings.get("kind", "structured")
    modules = ("mesh_checkpoint", "geometry")
    packages = ("numpy", "scikit-fem")
    if kind == "gmsh":
        modules += ("perforated",)
        packages += ("gmsh", "meshio")
    sources = [f"src/phononic/{module}_AP_pnc.py" for module in modules]
    return {
        "dimensions_m": [geometry.length_m, geometry.width_m, geometry.thickness_m],
        "pattern": {key: pattern[key] for key in ("source_sha256", "holes")} if pattern is not None else None,
        "settings": {**settings, "kind": kind},
        "source_sha256": {key: implementation["source_sha256"][key] for key in sources},
        "packages": {key: implementation["packages"][key] for key in packages},
        "python": implementation["python"],
        "platform": implementation["platform"],
    }


def build_mesh(geometry, holes, settings: dict, max_port_dofs: int | None) -> tuple[MeshTet, dict]:
    if settings.get("kind", "structured") == "gmsh":
        from .perforated_AP_pnc import make_perforated_mesh, volume_diagnostics

        mesh = make_perforated_mesh(
            geometry, holes, settings["mesh_size_nm"] * 1e-9,
            port_mesh_size_m=settings["port_size_nm"] * 1e-9,
            circle_points=settings["circle_resolution"], max_port_dofs=max_port_dofs,
            element_order=settings["element_order"],
        )
        diagnostics = {"geometry_diagnostics": volume_diagnostics(mesh, geometry, holes)}
    else:
        mesh = make_straight_mesh(geometry, settings["cells"])
        diagnostics = {}
    boundary = mesh.boundaries
    order = settings["element_order"]
    counts = {}
    for face in ("xleft", "xright"):
        triangles = mesh.facets[:, boundary[face]]
        nodes = np.unique(triangles).size
        if order == 2:
            edges = np.hstack((triangles[[0, 1]], triangles[[0, 2]], triangles[[1, 2]]))
            nodes += np.unique(np.sort(edges, axis=0), axis=1).shape[1]
        counts[face] = 3 * nodes
    return mesh, {**diagnostics, "ndof": 3 * (int(mesh.nvertices) + (mesh.edges.shape[1] if order == 2 else 0)),
                  "mesh_vertices": int(mesh.nvertices), "mesh_tetrahedra": int(mesh.nelements),
                  "boundary_facet_counts": {name: len(facets) for name, facets in boundary.items()},
                  "end_face_dofs": counts}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(directory: Path, mesh_hash: str) -> tuple[MeshTet, dict]:
    record_path = directory / "summary_AP_pnc.json"
    mesh_path = directory / "mesh_AP_pnc.npz"
    try:
        record = json.loads(record_path.read_bytes())
        serialized_inputs = json.dumps(record["mesh_inputs"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if (record["status"] != "mesh_ready" or record["mesh_hash"] != mesh_hash
                or hashlib.sha256(serialized_inputs).hexdigest() != mesh_hash
                or record["mesh_sha256"] != _sha(mesh_path)):
            raise ValueError("saved identity or mesh checksum differs")
        with np.load(mesh_path, allow_pickle=False) as arrays:
            mesh = MeshTet(arrays["points_m"], arrays["tetrahedra"]).with_boundaries(
                {name: arrays[name] for name in ("xleft", "xright", "free")}
            )
            if "solid" in arrays:
                mesh = mesh.with_subdomains({"solid": arrays["solid"]})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Mesh checkpoint changed or is invalid: {directory} ({exc})") from exc
    return mesh, record


def save_checkpoint(directory: Path, mesh: MeshTet, record: dict) -> tuple[MeshTet, dict, bool]:
    """Publish the completed mesh and manifest together, without replacing an existing checkpoint."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="mesh-", dir=directory.parent) as staging:
        staged = Path(staging)
        mesh_path = staged / "mesh_AP_pnc.npz"
        np.savez_compressed(mesh_path, points_m=mesh.p, tetrahedra=mesh.t,
                            **mesh.boundaries, **(mesh.subdomains or {}))
        record = {**record, "mesh_sha256": _sha(mesh_path)}
        (staged / "summary_AP_pnc.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        try:
            staged.rename(directory)
        except OSError:
            if not directory.is_dir():
                raise
            existing_mesh, existing_record = load_checkpoint(directory, record["mesh_hash"])
            return existing_mesh, existing_record, True
    return mesh, record, False
