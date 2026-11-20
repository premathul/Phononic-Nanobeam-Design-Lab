"""Load checked native inputs for the endpoint snapshot pilot. Author: Athul Prem | _AP_pnc."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools import map_scattering_AP_pnc as native
from tools import read_thin_device_grid_AP_pnc as grid
from tools.compare_scattering_AP_pnc import require


ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = Path("results/runs/physical_thickness_device_AP_pnc")
STUDY = ROOT / STUDY_PATH
TARGETS = grid.TARGETS
FREQUENCIES = grid.FREQUENCIES
SHARED = ("mesh", "stiffness", "mass", "port_section")
LAYOUT_FIELDS = ("points_m", "tetrahedra", "dof_coordinates_m", "dof_component", "element_dofs", "nodal_dofs")
PORT_FIELDS = (
    "port_displacements", "port_axial_forces", "port_wavenumbers_rad_m", "port_left_dofs", "port_right_dofs",
    "reflection_all_mode_indices", "transmission_all_mode_indices", "reflection_mode_indices",
    "transmission_mode_indices", "incident_mode_indices", "incident_powers_w", "frequencies_hz",
)


@dataclass(frozen=True)
class SnapshotPoint:
    frequency_hz: int
    fields_path: Path
    displacements: np.ndarray | None
    port_fields: dict[str, np.ndarray]
    reflection: np.ndarray
    transmission: np.ndarray
    reference_drives: tuple[int, ...]
    evidence: dict

    @property
    def incident_roots(self):
        return self.port_fields["incident_mode_indices"]

    @property
    def reflection_roots(self):
        return self.port_fields["reflection_mode_indices"]

    @property
    def transmission_roots(self):
        return self.port_fields["transmission_mode_indices"]


@dataclass(frozen=True)
class SnapshotLevel:
    central_max_edge_nm: float
    body_dofs: int
    preparation: dict[str, Path]
    endpoints: tuple[SnapshotPoint, SnapshotPoint]
    midpoint: SnapshotPoint
    evidence: dict


def _read_preparation(directory, pin, expected, level, root):
    require(native._verified(pin, root) == directory / native.CHECKPOINT,
            f"Checkpoint path differs from the published map in {directory}.")
    state = native.read_map_checkpoint(directory)
    shared = state["shared"]
    require(shared["mesh_hash"] == level["mesh_hash"]
            and shared["reference_planes_x_m"] == grid.PLANES,
            f"The volume mesh identity or reference planes changed in {directory}.")
    paths = {name: native._verified(shared["artifacts"][name], directory) for name in SHARED}
    actual = {key: {**shared["artifacts"][key], "path": path.relative_to(root).as_posix()}
              for key, path in paths.items()}
    require(actual == expected, f"Mesh, stiffness, mass or SAFE preparation differs in {directory}.")
    return state, paths


def _volume_layout(mesh, body_dofs):
    element = native.ElementVector(native.ElementTetP2())
    basis = native._DisplacementDofLayout(mesh, element, native.Dofs(mesh, element), 4)
    require(basis.N == body_dofs, "The checked mesh and quadratic displacement DOF count differ.")
    components = np.empty(basis.N, dtype=np.int8)
    for component, indices in enumerate(basis.split_indices()):
        components[indices] = component
    return dict(zip(LAYOUT_FIELDS,
                    (mesh.p.T, mesh.t.T, basis.doflocs.T, components, basis.element_dofs, basis.nodal_dofs), strict=True))


def _native_blocks(table, fields, arrays):
    incoming = fields["incident_mode_indices"].tolist()
    drives_by_root = {row["incident_port_index"]: row["reference_drive"] for row in table["drives"]}
    require(len(incoming) == 5 and set(incoming) == drives_by_root.keys(),
            "The native incident roots differ from the checked coefficient table.")
    raw = {(row["kind"], row["outgoing_port_index"], row["incident_port_index"]):
           complex(row["amplitude_real"], row["amplitude_imag"]) for row in table["coefficients"]}
    blocks = {}
    for name in grid.KINDS:
        outgoing = fields[f"{name}_mode_indices"].tolist()
        require(len(set(outgoing)) == len(outgoing) == 5, f"Invalid native {name} root inventory.")
        matrix = np.array([[raw[name, row, column] for column in incoming] for row in outgoing])
        require(np.array_equal(matrix, arrays[f"{name}_amplitudes"]),
                f"The {name} block in the checked CSV and field archive differs.")
        blocks[name] = matrix
    return blocks, tuple(drives_by_root[root] for root in incoming)


def _read_point(root, directory, state, sample, table, csv_pin, layout, traces, *, endpoint):
    frequency = table["frequency_hz"]
    pin = sample["archive"]
    point, = (row for row in state["points"] if row["frequency_GHz"] * 1e9 == frequency)
    summary_path = native._verified(pin["point_summary"], root)
    require(point["status"] == "accepted" and native._verified(point["accepted"]["summary"], directory) == summary_path,
            f"The native checkpoint does not identify the accepted point at {frequency} Hz.")
    summary = native._read(summary_path)
    provenance = summary["checkpointed_scattering"]
    require(summary["configuration"] == native._point_config(state["inputs"]["configuration"], frequency / 1e9)
            and summary["mesh_checkpoint"]["mesh_hash"] == state["shared"]["mesh_hash"]
            and summary["mesh_checkpoint"]["mesh_sha256"] == state["shared"]["artifacts"]["mesh"]["sha256"]
            and summary["frequency_hz"] == frequency and summary["incident_side"] == "left"
            and summary["power_balance_passed"] is True
            and provenance["shared_sha256"] == native.workflow.digest(state["shared"])
            and provenance["reference_planes_x_m"] == grid.PLANES
            and all(provenance["original_equations"][key] == value
                    for key, value in sample["original_equations"].items()),
            f"The saved point does not bind the checked preparation and original equations at {frequency} Hz.")
    path = native._verified(pin[native.FIELDS], root)
    require(path == summary_path.with_name(native.FIELDS)
            and summary["output_sha256"][native.FIELDS] == pin[native.FIELDS]["sha256"],
            f"The selected fields do not belong to the accepted solution at {frequency} Hz.")
    with np.load(path, allow_pickle=False) as arrays:
        require(arrays["element_order"].item() == 2
                and all(np.array_equal(arrays[key], value) for key, value in layout.items())
                and all(np.array_equal(arrays[f"port_{side}_dofs"], dofs) for side, dofs in traces.items()),
                f"The ordered volume or port DOFs changed at {frequency} Hz.")
        fields = {key: arrays[key] for key in PORT_FIELDS}
        require(np.array_equal(fields["frequencies_hz"], np.full(5, frequency)),
                f"The archived displacement columns do not belong to {frequency} Hz.")
        blocks, drives = _native_blocks(table, fields, arrays)
        displacements = arrays["displacements_per_sqrt_w"] if endpoint else None
        if endpoint:
            require(displacements.shape == (len(layout["dof_component"]), 5)
                    and np.iscomplexobj(displacements) and np.isfinite(displacements).all(),
                    f"The five endpoint displacement snapshots are invalid at {frequency} Hz.")
    evidence = {**sample, "native_coefficients": csv_pin, "reference_drives_in_native_order": drives}
    return SnapshotPoint(frequency, path, displacements, fields, blocks["reflection"], blocks["transmission"],
                         drives, evidence)


def load_snapshot_grid(target_nm, study=STUDY):
    """Verify one archived mesh level and load only its two endpoint volume fields."""
    study = Path(study).resolve()
    require(study.parts[-len(STUDY_PATH.parts):] == STUDY_PATH.parts,
            f"The checked archive must retain its original relative directory {STUDY_PATH.as_posix()}.")
    require(target_nm in TARGETS, f"No saved thin-device grid at {target_nm} nm.")
    root = study.parents[len(STUDY_PATH.parts) - 1]
    compact = grid.load_thin_grid(root)
    require(compact is not None, f"The pinned nine-point comparison is missing under {root}.")
    index = TARGETS.index(target_nm)
    table_points = compact["levels"][index]["points"]
    sources = {row["path"]: {**row, "path": Path(row["path"]).relative_to(root).as_posix()}
               for row in compact["sources"]}
    comparison = native._read(Path(compact["source_comparison"]))
    level = comparison["mesh_levels"][index]
    pins = level["shared_inputs_by_map"]
    preparations = {}
    for sample, table in zip(level["samples"], table_points, strict=True):
        directory = Path(table["source_map"]).parent
        require(directory.parent == study, f"Unexpected checked map directory {directory}.")
        if directory not in preparations:
            preparations[directory] = _read_preparation(directory, sample["archive"]["checkpoint"],
                                                        pins[directory.name], level, root)
    directory = Path(table_points[-1]["source_map"]).parent
    state, paths = preparations[directory]
    for name in SHARED:
        identities = {(row[name]["sha256"], row[name]["bytes"]) for row in pins.values()}
        require(len(identities) == 1, f"Same-mesh frequency points have different {name} byte identities.")
    mesh_summary = native._verified(state["shared"]["artifacts"]["mesh_summary"], directory)
    require(paths["mesh"] == mesh_summary.with_name("mesh_AP_pnc.npz"), "The checked mesh manifest points elsewhere.")
    mesh, manifest = native.workflow.load_checkpoint(mesh_summary.parent, level["mesh_hash"])
    require(manifest["ndof"] == level["body_dofs"], "The checked mesh and published body DOF count differ.")
    layout = _volume_layout(mesh, level["body_dofs"])
    with np.load(paths["port_section"], allow_pickle=False) as arrays:
        traces = {side: arrays[f"{side}_dofs"] for side in ("left", "right")}
    for dofs, plane in zip(traces.values(), grid.PLANES, strict=True):
        coords = layout["dof_coordinates_m"]
        require(dofs.ndim == 1 and dofs.dtype.kind in "iu" and len(set(dofs)) == len(dofs)
                and dofs.min() >= 0 and dofs.max() < len(coords) and np.all(coords[dofs, 0] == plane),
                "A native port trace differs from the ordered body DOFs on its original plane.")
    points = []
    for sample, table in zip(level["samples"], table_points, strict=True):
        directory = Path(table["source_map"]).parent
        point_state, _ = preparations[directory]
        for key in ("geometry", "material", "mesh"):
            require(point_state["inputs"]["configuration"][key] == state["inputs"]["configuration"][key],
                    f"The fixed {key} changes across frequencies on the {target_nm:.8g} nm mesh.")
        require(point_state["inputs"]["hole_pattern"] == state["inputs"]["hole_pattern"],
                f"The physical hole pattern changes across frequencies on the {target_nm:.8g} nm mesh.")
        points.append(_read_point(root, directory, point_state, sample, table, sources[table["source_coefficients"]],
                                  layout, traces, endpoint=sample["frequency_hz"] != FREQUENCIES[1]))
    evidence = {"comparison": grid.COMPARISON, "mesh_hash": level["mesh_hash"],
                "same_mesh_shared_inputs_by_map": pins, "reference_planes_x_m": grid.PLANES,
                "volume_dof_layout_sha256": native._array_digest(*layout.values())}
    return SnapshotLevel(target_nm, level["body_dofs"], paths, (points[0], points[2]), points[1], evidence)
