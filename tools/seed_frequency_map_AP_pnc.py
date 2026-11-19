#!/usr/bin/env python3
"""Reuse verified preparation in a separate native frequency map. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import math
import os
from pathlib import Path
from shutil import copyfile
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools import map_scattering_AP_pnc as native

workflow = native.workflow


def _verify_donor(donor: Path) -> tuple[dict, dict[str, Path], dict]:
    checkpoint = donor / native.CHECKPOINT
    if not checkpoint.is_file() or checkpoint.resolve() != checkpoint:
        raise ValueError("Choose a native donor checkpoint in its own directory. Imported maps are read-only.")
    checkpoint_sha = native._sha(checkpoint)
    state = native.read_map_checkpoint(donor)

    inputs = state["inputs"]
    config = inputs["configuration"]
    settings = json.loads(json.dumps(workflow.resolve_config(config), allow_nan=False))
    geometry, holes, pattern = workflow._resolve_geometry(settings)
    implementation = workflow.source_identity()
    identity = native._numerical_identity(implementation)
    identity["source_sha256"] = {**identity["source_sha256"], **{
        path.relative_to(ROOT).as_posix(): native._sha(path)
        for path in (Path(native.__file__).resolve(), Path(native.comparison.__file__).resolve())}}
    threads = {key: os.getenv(key) for key in native.THREADS}
    if (workflow.digest(settings) != workflow.digest(config) or pattern != inputs["hole_pattern"]
            or identity != inputs["numerical_identity"] or threads != inputs["thread_environment"]):
        raise ValueError("Donor configuration, hole source, numerical source or environment changed.")

    shared = state["shared"]
    if not shared:
        raise ValueError("The native donor has no saved preparation.")
    mesh_dir, mesh_hash, mesh_inputs = workflow._mesh_location(Path(), geometry, pattern, config["mesh"], identity)
    artifacts = shared["artifacts"]
    filenames = (mesh_dir / native.RUN_SUMMARY, mesh_dir / "mesh_AP_pnc.npz", Path("K.npz"), Path("M.npz"),
                 Path("port_section_AP_pnc.npz"))
    stage = Path(artifacts["mass"]["path"]).parent
    if (shared["mesh_hash"] != mesh_hash or shared["reference_planes_x_m"] != [0., geometry.length_m]
            or set(artifacts) != set(native.SHARED_FILES) or len(stage.parts) != 1
            or not stage.name.startswith("shared-")
            or any(artifacts[name]["path"] != (stage / filename).as_posix()
                   for name, filename in zip(native.SHARED_FILES, filenames, strict=True))):
        raise ValueError("Donor shared preparation layout or mesh identity changed.")
    paths = {name: native._verified(artifacts[name], donor) for name in native.SHARED_FILES}
    manifest = native._read(paths["mesh_summary"])
    if (manifest["status"] != "mesh_ready" or manifest["mesh_hash"] != mesh_hash
            or workflow.digest(manifest["mesh_inputs"]) != workflow.digest(mesh_inputs)
            or manifest["mesh_sha256"] != artifacts["mesh"]["sha256"]
            or paths["mesh"] != paths["mesh_summary"].parent / "mesh_AP_pnc.npz"):
        raise ValueError("Donor mesh manifest differs from its configured inputs or mesh checksum.")
    accepted = [point for point in state["points"] if point["status"] == "accepted"]
    for point in accepted:
        # Accepted summaries pin the exact shared record as well as the original equation and flux gates.
        _, summary = native._read_accepted(point, config, shared, donor)
        if (native._numerical_identity(summary["implementation"]) != native._numerical_identity(implementation)
                or summary["hole_pattern_provenance"] != manifest["hole_pattern_provenance"]
                or summary["hole_count"] != len(holes)):
            raise ValueError("Donor accepted response differs from the recorded numerical inputs.")
    if not accepted:
        # A preparation-only publication pins the raw checkpoint before any point exists.
        published = donor / native.MAP
        if not published.is_file() or published.resolve() != published:
            raise ValueError("The donor needs a published map or an accepted response to verify shared preparation.")
        result = native.read_saved_map(donor)
        if (result["checkpoint_sha256"] != checkpoint_sha
                or any(result[key] != state[key] for key in ("inputs_sha256", "inputs", "shared", "points"))):
            raise ValueError("Published donor preparation differs from its checkpoint.")
    if native._sha(checkpoint) != checkpoint_sha:
        raise ValueError("Donor checkpoint changed during verification.")
    provenance = {"donor_checkpoint_sha256": checkpoint_sha, "donor_shared_sha256": workflow.digest(shared),
                  "donor_source_git_commit": state["source_git_commit"],
                  "seeder_source_sha256": native._sha(Path(__file__).resolve())}
    return {"inputs": inputs, "shared": shared, "source_git_commit": implementation["git_commit"]}, paths, provenance


def _install_preparation(output: Path, state: dict, donor_paths: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".frequency-seed-", dir=output.parent) as scratch:
        staged = Path(scratch).resolve() / "map"
        for name, artifact in state["shared"]["artifacts"].items():
            target = staged / artifact["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            copyfile(donor_paths[name], target)
            native._verified(artifact, staged)
        native._write_checkpoint(staged / native.CHECKPOINT, state)
        try:
            if output.is_dir() and not any(output.iterdir()):
                output.rmdir()
            staged.rename(output)
        except OSError:
            # A competing initializer may have installed the checkpoint first. Verify it before use.
            if not (output / native.CHECKPOINT).is_file():
                raise


def seed_frequency_map(donor: Path, output: Path, frequencies_ghz: Sequence[float], *, max_new_points: int = 1) -> dict:
    """Copy intact preparation, then use the normal native map runner and point gates."""
    frequencies = list(frequencies_ghz)
    if (not frequencies or any(not workflow._finite_number(f) or f <= 0 or not math.isfinite(f * 1e9) for f in frequencies)
            or any(b * 1e9 <= a * 1e9 for a, b in zip(frequencies, frequencies[1:]))):
        raise ValueError("Supply a nonempty, strictly increasing list of positive frequencies in GHz.")
    if type(max_new_points) is not int or max_new_points < 0:
        raise ValueError("max_new_points must be a nonnegative integer.")
    donor, output = Path(donor).resolve(strict=True), Path(output).resolve()
    if output.is_relative_to(donor) or donor.is_relative_to(output):
        raise ValueError("The output and donor must be separate map directories.")
    initialize = not output.exists() or (output.is_dir() and not any(output.iterdir()))
    if not initialize and not (output / native.CHECKPOINT).is_file():
        raise ValueError("Select an empty output directory or the same seeded frequency map.")

    source, paths, provenance = _verify_donor(donor)
    # Keep the donor's fixed configuration anchor for normal native resume. The runner sets each point frequency.
    inputs = {**source["inputs"], "frequencies_GHz": frequencies}
    state = {**native.AUTHOR, **native.CONVERGENCE, "schema_version": 1,
             "inputs_sha256": workflow.digest(inputs), "inputs": inputs,
             "source_git_commit": source["source_git_commit"], "shared": source["shared"],
             "preparation_reuse": provenance,
             "points": [{"index": index, "frequency_GHz": frequency, "status": "pending", "attempts": [],
                         **native.CONVERGENCE} for index, frequency in enumerate(frequencies, 1)]}
    if initialize:
        _install_preparation(output, state, paths)
    if any(path.resolve() != path for path in output.iterdir()):
        raise ValueError("The output contains an entry linked outside its native path.")
    saved = native.read_map_checkpoint(output)
    prior = saved.get("preparation_reuse")
    donor_keys = provenance.keys() - {"seeder_source_sha256"}
    if (saved["inputs_sha256"] != state["inputs_sha256"] or saved["shared"] != state["shared"]
            or not isinstance(prior, dict)
            or not native.comparison.is_sha256(prior.get("seeder_source_sha256"))
            or any(prior.get(key) != provenance[key] for key in donor_keys)):
        raise ValueError("Existing map differs from the verified donor or requested frequencies.")
    return native.map_scattering(inputs["configuration"], frequencies, output, max_new_points=max_new_points)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-map", required=True, type=Path, dest="donor", help="Native donor map with saved preparation.")
    parser.add_argument("--output", required=True, type=Path, help="New or matching seeded map directory.")
    parser.add_argument("--frequencies-ghz", required=True, type=float, nargs="+",
                        help="Strictly increasing frequency samples in GHz.")
    parser.add_argument("--max-new-points", type=int, default=1,
                        help="Maximum new point attempts (default 1). Use 0 to save preparation only.")
    args = parser.parse_args(argv)
    try:
        result = seed_frequency_map(args.donor, args.output, args.frequencies_ghz, max_new_points=args.max_new_points)
    except KeyboardInterrupt:
        print("Frequency map interrupted. Reuse the same inputs and output directory to resume.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"Frequency map could not complete: {exc}", file=sys.stderr)
        return 2
    accepted = sum(point["status"] == "accepted" for point in result["points"])
    print(f"Accepted {accepted}/{len(result['points'])} frequencies. Report: {(args.output / native.REPORT).resolve()}")
    return 2 if any(point["status"] == "failed" for point in result["points"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
