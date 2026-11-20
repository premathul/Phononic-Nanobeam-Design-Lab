"""Prepare or resume one goal-selected native 1 GHz control. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    os.environ.update(dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                     "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"))
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools import central_patch_refinement_AP_pnc as nested
from tools import goal_patch_refinement_AP_pnc as goal
from tools import run_taper_patch_response_AP_pnc as taper

native, point_runner, workflow = taper.native, taper.single_point, taper.workflow
require = taper.require
FREQUENCY = point_runner.FREQUENCY_GHZ
RUN_NAMES = {"cavity_13_14": "thin100_cavity_pair225to168p75_1GHz",
             "outer_taper_09_18": "thin100_outer_taper_pair225to168p75_1GHz"}
SECTION_KEYS = tuple(name for name in point_runner.SECTION_ARRAYS if name not in ("left_dofs", "right_dofs"))


def _identity():
    threads = {name: os.getenv(name) for name in native.THREADS}
    require(threads == native.THREADS, "Use one numerical thread.")
    revision = workflow.source_identity()
    identity = native._numerical_identity(revision)
    modules = (goal, nested, taper, native, point_runner, native.comparison)
    for module_path in (Path(__file__), *(Path(module.__file__) for module in modules)):
        path = module_path.resolve()
        identity["source_sha256"][path.relative_to(ROOT).as_posix()] = native._sha(path)
    return revision["git_commit"], identity, threads


def _load_donors(coarse, fine):
    coarse_mesh, fine_mesh, source = nested.load_nested_patch_sources(coarse, fine)
    states = [native.read_map_checkpoint(directory) for directory in (coarse, fine)]
    settings = [state["inputs"]["configuration"] for state in states]
    config = settings[0]
    pattern = states[0]["inputs"]["hole_pattern"]
    require(config["geometry"]["thickness_nm"] == 100
            and tuple(item["mesh"]["central_max_edge_nm"] for item in settings) == (225, 168.75)
            and [config["mesh"][name] for name in ("mesh_size_nm", "lead_mesh_size_nm", "circle_resolution",
                                                  "thickness_layers")] == [400, 150, 32, 2]
            and config["analysis"]["incident_side"] == "left"
            and config["analysis"]["frequency_GHz"] == FREQUENCY
            and config["resources"].get("linear_solver", "direct") == "direct",
            "Choose the accepted 100 nm, two-layer, 32-rim, 225/168.75 nm nested left-incident 1 GHz donors.")
    sections = []
    for role, directory, state in zip(("coarse", "fine"), (coarse, fine), states, strict=True):
        accepted = [point for point in state["points"]
                    if point["frequency_GHz"] == FREQUENCY and point["status"] == "accepted"]
        require(len(accepted) == 1 and state["checkpoint_data_sha256"] == native._read(
            native._verified(source[role]["checkpoint"], directory))["checkpoint_data_sha256"],
            "Each donor must retain the accepted 1 GHz point and its original checkpoint.")
        section = state["shared"]["artifacts"]["port_section"]
        sections.append(native._verified(section, directory))
        source[role].update(port_section=section, accepted_point_summary=accepted[0]["accepted"]["summary"])
    return coarse_mesh, fine_mesh, source, config, pattern, sections


def _check_sections(shared, sections, output):
    # Body DOF IDs can move. SAFE section order and operators must be exact.
    candidate_path = native._verified(shared.record["artifacts"]["port_section"], output)
    with np.load(candidate_path, allow_pickle=False) as candidate:
        for path in sections:
            with np.load(path, allow_pickle=False) as donor:
                require(all(np.array_equal(candidate[key], donor[key]) for key in SECTION_KEYS),
                        "The selected control changed a native end section or its SAFE operator.")


def _prepare(output, region, mesh, donors, config, pattern, checks, sections, identity, threads, revision):
    selection = {key: checks["goal_parent_partition"][key] for key in (
        "index_order", "hole_bins", "hole_bin_edges_x_m", "selected_parent_indices", "replaced_parent_indices")}
    selection["source_sha256"] = {path: identity["source_sha256"][path] for path in (
        Path(__file__).resolve().relative_to(ROOT).as_posix(), Path(goal.__file__).resolve().relative_to(ROOT).as_posix())}
    source = {**donors, "goal_patch_selection": selection}
    label = RUN_NAMES[region].removesuffix("225to168p75_1GHz")
    config = workflow.resolve_config({**config, "name": f"{label}_AP_pnc",
                                      "resources": {**config["resources"], "max_dofs": checks["ndof"]}})
    shared, pattern = taper._prepare(output, config, pattern, identity, mesh, checks, source, region)
    _check_sections(shared, sections, output)
    inputs = {"configuration": config, "hole_pattern": pattern, "frequencies_GHz": [FREQUENCY],
              "numerical_identity": identity, "thread_environment": threads}
    state = {**native.AUTHOR, **native.CONVERGENCE, "schema_version": 1, "inputs": inputs,
             "inputs_sha256": workflow.digest(inputs), "source_git_commit": revision,
             "goal_patch_donors": donors, "goal_patch_region": region, "original_end_sections_match_exactly": True,
             "shared": shared.record, "points": [{"index": 1, "frequency_GHz": FREQUENCY, "status": "pending",
                                                   "attempts": [], **native.CONVERGENCE}]}
    native._write_checkpoint(output / native.CHECKPOINT, state)
    return state, shared


def run(coarse, fine, output, *, region, prepare_only=False):
    """Run or resume one bounded native point with an immutable goal selection."""
    coarse, fine, output = (Path(p).resolve() for p in (coarse, fine, output))
    require(all(not output.is_relative_to(donor) and not donor.is_relative_to(output) for donor in (coarse, fine)),
            "Choose an output separate from both mesh donors.")
    require(region in RUN_NAMES and region in goal.REGIONS, "Choose one of the two goal-selected hole pairs.")
    revision, identity, threads = _identity()
    coarse_mesh, fine_mesh, donors, config, pattern, sections = _load_donors(coarse, fine)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / native.CHECKPOINT
    with native._exclusive(output / ".frequency_map.lock"):
        if checkpoint.exists():
            state = native.read_map_checkpoint(output)
            state.pop("checkpoint_data_sha256")
            require(state.get("goal_patch_donors") == donors and state.get("goal_patch_region") == region
                    and state.get("original_end_sections_match_exactly") is True
                    and state["inputs"]["frequencies_GHz"] == [FREQUENCY]
                    and state["inputs"]["numerical_identity"] == identity
                    and state["inputs"]["thread_environment"] == threads,
                    "The existing checkpoint uses a different goal selection, donors, or environment.")
            point, = state["points"]
            if point["status"] == "accepted":
                accepted = {1: native._read_accepted(point, state["inputs"]["configuration"], state["shared"], output)}
                if (output / native.MAP).exists():
                    publication = native.read_saved_map(output)
                    if publication["checkpoint_sha256"] == native._sha(checkpoint):
                        return publication
                return native._publish(state, None, accepted, output)
            if prepare_only:
                for artifact in state["shared"]["artifacts"].values():
                    native._verified(artifact, output)
                return native._publish(state, None, {}, output)
            shared = native._restore_shared(state["shared"], state["inputs"]["configuration"], output)
            _check_sections(shared, sections, output)
        else:
            require(all(p.name == ".frequency_map.lock" or (p.name.startswith("shared-") and p.is_dir())
                        for p in output.iterdir()), "Output contains a different study.")
            centers = tuple(hole["x_m"] for hole in pattern["holes"])
            mesh, checks = goal.make_goal_patch_control(coarse_mesh, fine_mesh,
                                                        tuple(donors["nominal_mirror_cuts_x_m"]), centers, region=region)
            state, shared = _prepare(output, region, mesh, donors, config, pattern, checks, sections,
                                     identity, threads, revision)
            if prepare_only:
                return native._publish(state, shared, {}, output)
        accepted = point_runner._run_or_recover(state, output, shared, identity)
        return native._publish(state, shared, accepted, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--region", choices=RUN_NAMES, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.coarse, args.fine, args.output, region=args.region, prepare_only=args.prepare_only)
    point, = result["points"]
    print(f"{point['status']}: {args.output.resolve()}")
    if point["status"] == "failed":
        print(point["attempts"][-1])
    return 0 if point["status"] == "accepted" or (args.prepare_only and point["status"] == "pending") else 1


if __name__ == "__main__":
    raise SystemExit(main())
