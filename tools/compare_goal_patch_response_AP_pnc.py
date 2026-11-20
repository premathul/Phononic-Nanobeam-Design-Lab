"""Test parent-work predictions against saved 1 GHz pair refinements. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import json
from math import fsum
from pathlib import Path
import runpy
import sys
from tempfile import NamedTemporaryFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools import compare_taper_patch_response_AP_pnc as taper
from tools import goal_patch_refinement_AP_pnc as selector
from tools import run_goal_patch_response_AP_pnc as runner


native, require = taper.native, taper.require
WORK = ROOT / "verification/regional_reflection_work_1GHz_AP_pnc.json"
OUTPUT = ROOT / "verification/goal_patch_response_1GHz_AP_pnc.json"
WORK_PIN_KEYS = {"checkpoint": "checkpoint", "point": "point_summary", "mesh": "mesh",
                 "section": "port_section", "fields": native.FIELDS, "flux": "independent_flux_replay"}
SELECTION_KEYS = ("index_order", "hole_bins", "hole_bin_edges_x_m", "selected_parent_indices",
                  "replaced_parent_indices")
SIGNIFICANT_PP = 1e-7
BODY_DOF_UNIT = 1000
BODY_DOF_SUFFIX = "_per_1000_added_body_dofs"


def _indices(values, label):
    ids = np.asarray(values)
    require(ids.ndim == 1 and ids.dtype.kind in "iu" and len(ids) > 0
            and ids[0] >= 0 and np.all(ids[1:] > ids[:-1]), f"Invalid ordered {label}.")
    return ids


def _parent_predictions(parent_work, selection):
    indices = _indices(parent_work["indices"], "parent work indices")
    actual = _indices(selection["replaced_parent_indices"], "replaced parent indices")
    selected = _indices(selection["selected_parent_indices"], "selected parent indices")
    holes = np.asarray(parent_work["hole_numbers"])
    work = np.asarray(parent_work["delta_T_percentage_points_by_drive"], dtype=float)
    require(holes.shape == indices.shape and holes.dtype.kind in "iu" and np.all((holes >= 1) & (holes <= 26))
            and work.shape == (len(indices), 5) and np.isfinite(work).all(), "Invalid five-drive parent work.")
    full_bin = np.isin(holes, selection["hole_bins"])
    applied = np.isin(indices, actual)
    require(np.isin(actual, selected).all() and np.count_nonzero(applied) == len(actual)
            and np.all(full_bin[applied]), "Replaced parents must be selected, present in the work and in the declared bins.")
    unapplied = full_bin & ~applied
    counts = {"selected_parent_count": len(selected), "replaced_parent_count": len(actual),
              "full_bin_work_parent_count": int(np.count_nonzero(full_bin)),
              "full_bin_unreplaced_parent_count": int(np.count_nonzero(unapplied)),
              "full_bin_unreplaced_parent_indices": indices[unapplied].tolist()}
    return work[applied].sum(axis=0), work[full_bin].sum(axis=0), work[unapplied].sum(axis=0), counts


def _read_goal(study, region, prior, reader, baseline, donors, donor_meshes):
    name = runner.RUN_NAMES[region]
    directory = (study / name).resolve()
    flux_path = (study / f"{name}_1GHz_flux_AP_pnc.json").resolve()
    require(directory.parent == flux_path.parent == study, f"Unexpected archive location for {region}.")
    checkpoint = native.read_map_checkpoint(directory)
    record, config, pattern = reader(directory, flux_path, taper.FREQUENCY_GHZ)
    shared = checkpoint["shared"]
    manifest_path = native._verified(shared["artifacts"]["mesh_summary"], directory)
    mesh, manifest = native.workflow.load_checkpoint(manifest_path.parent, shared["mesh_hash"])
    diag = manifest["geometry_diagnostics"]["taper_patch_refinement"]
    control = pattern["taper_patch_refinement"]
    selection = diag["goal_parent_partition"]
    source = {module: native._link(Path(module.__file__).resolve(), ROOT) for module in (runner, selector)}
    selection_sha = {pin["path"]: pin["sha256"] for pin in source.values()}
    config_name = name.removesuffix("225to168p75_1GHz") + "_AP_pnc"
    require(
        config == {**baseline.config, "name": config_name,
                   "resources": {**baseline.config["resources"], "max_dofs": diag["ndof"]}}
        and record["physical_thickness_nm"] == 100
        and record["reference_planes_x_m"] == prior["reference_planes_x_m"]
        and checkpoint["inputs"]["frequencies_GHz"] == [taper.FREQUENCY_GHZ]
        and len(checkpoint["points"]) == 1 and checkpoint["points"][0]["index"] == 1
        and "preparation_reuse" not in checkpoint
        and checkpoint["goal_patch_region"] == control["refined_region"] == diag["region"] == region
        and checkpoint["goal_patch_donors"] == donors
        and checkpoint["original_end_sections_match_exactly"] is True
        and {key: value for key, value in pattern.items() if key != "taper_patch_refinement"} == baseline.pattern
        and manifest["hole_pattern_provenance"] == pattern
        and manifest["mesh_inputs"]["taper_patch_refinement"] == control
        and control["donors"] == {**donors, "goal_patch_selection": {
            **{key: selection[key] for key in SELECTION_KEYS}, "source_sha256": selection_sha}}
        and all(checkpoint["inputs"]["numerical_identity"]["source_sha256"][key] == sha
                for key, sha in selection_sha.items())
        and diag["control_source_sha256"] == source[selector]["sha256"]
        and record["body_dofs"] == manifest["ndof"] == diag["ndof"]
        and record["archive"]["mesh"]["sha256"] == manifest["mesh_sha256"]
        and native._array_digest(mesh.p, mesh.t) == control["mesh_geometry_connectivity_sha256"]
        == diag["control_coordinates_and_connectivity_sha256"],
        f"The accepted {region} map does not bind the exact selected geometry and solver inputs.",
    )
    expected_mesh, expected_report = selector.make_goal_patch_control(
        *donor_meshes, tuple(donors["nominal_mirror_cuts_x_m"]),
        [hole["x_m"] for hole in baseline.pattern["holes"]], region=region)
    require(selection == expected_report["goal_parent_partition"]
            and np.array_equal(mesh.p, expected_mesh.p) and np.array_equal(mesh.t, expected_mesh.t)
            and taper._surface_signature(mesh) == taper._surface_signature(baseline.mesh),
            f"The verified {region} mesh differs from its native coarse-to-fine parent replacements.")

    flux = native._read(flux_path)
    require([flux[key] for key in ("replay_source_sha256", "regional_replay_source_sha256")]
            == [pin["sha256"] for pin in prior["provenance"]["independent_flux_sources"]],
            f"The independent {region} flux producers differ from the published controls.")
    run_dir, summary, fixed, _, section_path = taper.nested._channel_input(record)
    field_path = native._verified(record["archive"][native.FIELDS], ROOT)
    section, trace = taper._read_section_and_trace(section_path, field_path, mesh, record["reference_planes_x_m"],
                                                   record["port_dofs"])
    require(all(np.array_equal(section[key], baseline.section[key]) for key in runner.SECTION_KEYS),
            f"The {region} native section changed.")
    data = taper.scattering.read_scattering(run_dir, summary, fixed)
    eigen = taper.layer._transmission_eigenchannels(record, data, flux, prior["original_numerical_limits"])
    reflected = np.linalg.svd(data[2]["reflection"], compute_uv=False)**2
    require(abs(reflected.sum() - sum(row["reflection"] for row in record["native_drives"]))
            < taper.layer.NATIVE_POWER_ROUNDOFF, f"The {region} reflection totals differ from the native block.")
    identity = checkpoint["inputs"]["numerical_identity"]
    record.update(point_index=1, platform={"assembly": identity["platform"], "solve": identity["platform"]},
                  original_equation_identity={key: flux["original_equations_saved"][key]
                                              for key in ("operator_csc_sha256", "rhs_sha256")})
    record["archive"]["mesh_summary"] = native._link(manifest_path, ROOT)
    return taper.Response(record, config, pattern, checkpoint, manifest, mesh, summary, fixed, section, trace,
                          data, eigen, reflected), selection


def _read_work(path, baseline, center, edge):
    bundle = native._read(path)
    case = bundle["cases"]["thin_first"]
    result = case["result"]
    require(bundle["frequency_hz"] == taper.FREQUENCY_GHZ * 1e9
            and result["original_planes_m"] == baseline.record["reference_planes_x_m"]
            and case["result_sha256"] == native.workflow.digest(result)
            and all(taper._same_pin(case["inputs"][side][key], response.record["archive"][archive_key])
                    for side, response in (("coarse", baseline), ("refined", center))
                    for key, archive_key in WORK_PIN_KEYS.items()),
            "The thin_first work record differs from the verified coarse and full-central responses.")
    work = result["parent_work"]
    ids = _indices(work["indices"], "thin_first parent indices")
    z, xy, triangles, owners = taper.patch._footprint(baseline.mesh)
    centers = np.array([hole["x_m"] for hole in baseline.pattern["holes"]])
    bins = np.searchsorted((centers[:-1] + centers[1:]) / 2, np.sort(xy[triangles, 0], axis=1).mean(axis=1)) + 1
    require(ids[-1] < len(triangles) and np.array_equal(bins[ids], work["hole_numbers"])
            and len(ids) == result["partition"]["changed_parent_count"], "thin_first parent ordinals or hole ownership differ.")
    coarse_cells, center_cells = (taper._tetrahedron_keys(case.mesh) for case in (baseline, center))
    reported = taper._tetrahedron_keys(baseline.mesh.restrict(np.flatnonzero(np.isin(owners, ids))))
    require(np.array_equal(reported, np.setdiff1d(coarse_cells, center_cells, assume_unique=True))
            and len(reported) == (len(z) - 1) * 3 * len(ids),
            "The parent work rows do not cover exactly the volume removed by the full central refinement.")
    roots = result["incident_roots"]
    incoming = edge["signed_mode_matches"]["incoming"]
    require(roots == [row["native_root_id"] for row in center.record["native_drives"]],
            "The work columns must be in the full-central native incident order.")
    order = [roots.index(root) for root in incoming["candidate_roots"]]
    values = np.asarray(work["delta_T_percentage_points_by_drive"], dtype=float)
    central = np.asarray(result["measured_delta_T_percentage_points"], dtype=float)
    end = np.asarray(result["end_correction_delta_T_percentage_points"], dtype=float)
    actual = np.asarray([row["candidate_minus_reference_percentage_points"]["transmission"]
                         for row in edge["drive_power_fractions"]])
    require(values.shape == (len(ids), 5) and central.shape == end.shape == actual.shape == (5,)
            and np.isfinite(values).all() and np.isfinite(central).all() and np.isfinite(end).all()
            and np.max(abs(central[order] - actual)) <= SIGNIFICANT_PP
            and np.max(abs(values.sum(axis=0) + end - central)) <= SIGNIFICANT_PP,
            "The reviewed parent contributions do not close against the native full-central increment.")
    return {**work, "delta_T_percentage_points_by_drive": values[:, order]}, roots


def _sign(value):
    return "positive" if value > SIGNIFICANT_PP else "negative" if value < -SIGNIFICANT_PP else "unresolved"


def _compare_prediction(work, selection, pair_edge, central_edge, *, added_body_dofs):
    require(type(added_body_dofs) is int and added_body_dofs > 0, "The pair refinement must add body DOFs.")
    predicted, full_bin, unreplaced, counts = _parent_predictions(work, selection)
    dof_scale = BODY_DOF_UNIT / added_body_dofs
    rows = []
    for index, (pair, central) in enumerate(zip(pair_edge["drive_power_fractions"],
                                               central_edge["drive_power_fractions"], strict=True)):
        same, selected, full = pair["reference"], pair["candidate"], central["candidate"]
        require(same == central["reference"] and same["drive"] == index + 1, "The coarse drive references differ.")
        coarse_t, actual_t, central_t = (item["transmission"] for item in (same, selected, full))
        measured = pair["candidate_minus_reference_percentage_points"]["transmission"]
        rows.append({
            "reference_drive": same["drive"], "coarse_native_root": same["native_root_id"],
            "pair_native_root": selected["native_root_id"], "full_central_native_root": full["native_root_id"],
            "coarse_transmission_fraction": coarse_t, "predicted_transmission_fraction": coarse_t + predicted[index] / 100,
            "measured_transmission_fraction": actual_t, "full_central_transmission_fraction": central_t,
            "prediction_percentage_points": float(predicted[index]), "measured_change_percentage_points": measured,
            f"prediction_percentage_points{BODY_DOF_SUFFIX}": float(predicted[index]) * dof_scale,
            f"measured_change_percentage_points{BODY_DOF_SUFFIX}": measured * dof_scale,
            "full_bin_work_percentage_points": float(full_bin[index]),
            "unreplaced_prediction_percentage_points": float(unreplaced[index]),
            "measured_minus_predicted_percentage_points": measured - float(predicted[index]),
            "full_central_minus_pair_percentage_points": 100 * (central_t - actual_t),
            "predicted_sign": _sign(predicted[index]), "measured_sign": _sign(measured),
            "sign_matches": _sign(predicted[index]) == _sign(measured) != "unresolved",
        })
    mean_fields = [key for key in rows[0] if key.endswith(("_fraction", "_percentage_points", BODY_DOF_SUFFIX))]
    return {"selection": {**selection, **counts}, "added_body_dofs": added_body_dofs, "per_drive": rows,
            "equal_incoherent_mean": {key: fsum(row[key] for row in rows) / len(rows) for key in mean_fields}}


def _per_dof_ranking_resolution(cases):
    return SIGNIFICANT_PP * BODY_DOF_UNIT / min(case["added_body_dofs"] for case in cases.values())


def _drive_checks(cases):
    report = []
    resolutions = (("", SIGNIFICANT_PP), (BODY_DOF_SUFFIX, _per_dof_ranking_resolution(cases)))
    for drive in (3, 5):
        rows = {region: next(row for row in case["per_drive"] if row["reference_drive"] == drive)
                for region, case in cases.items()}
        check = {"reference_drive": drive, "sign_matches_by_control": {region: row["sign_matches"]
                                                                     for region, row in rows.items()}}
        for suffix, resolution in resolutions:
            for name, quantity in (("signed_gain", float), ("magnitude", abs)):
                orders = {}
                for role, key in (("predicted", "prediction_percentage_points"),
                                  ("measured", "measured_change_percentage_points")):
                    ordered = sorted(rows, key=lambda region: quantity(rows[region][key + suffix]), reverse=True)
                    a, b = (quantity(rows[region][key + suffix]) for region in ordered)
                    orders[role] = ordered if abs(a - b) > resolution else None
                    check[f"{role}_order_by_{name}{suffix}"] = orders[role]
                check[f"{name}_ranking_matches{suffix}"] = (
                    orders["predicted"] == orders["measured"] if all(orders.values()) else None
                )
        report.append(check)
    return report


def replay(study=taper.STUDY, work_path=WORK):
    study, work_path = Path(study).resolve(), Path(work_path).resolve()
    require(study.is_relative_to(ROOT) and work_path.is_relative_to(ROOT),
            "Keep the native archives and work report under this checkout for portable evidence links.")
    prior = native._read(taper.PRIOR)
    reader_pin = prior["provenance"]["acceptance_reader"]
    reader = runpy.run_path(str(native._verified(reader_pin, ROOT)))["read_point"]
    baseline, center = (taper._read_response(study, role, prior, reader) for role in ("baseline", "taper_patch"))
    coarse, full, donors, _, _, _ = runner._load_donors(study / taper.MAPS["baseline"][0],
                                                       study / taper.MAPS["full_refinement"][0])
    central_edge = taper._edge(baseline, center)
    work_pin = native._link(work_path, ROOT)
    parent_work, work_roots = _read_work(native._verified(work_pin, ROOT), baseline, center, central_edge)
    cases = {}
    identities = {row.record["original_equation_identity"]["operator_csc_sha256"] for row in (baseline, center)}
    for region in runner.RUN_NAMES:
        candidate, selection = _read_goal(study, region, prior, reader, baseline, donors, (coarse, full))
        edge = taper._edge(baseline, candidate)
        identity = candidate.record["original_equation_identity"]["operator_csc_sha256"]
        require(identity not in identities, "The saved cases must have distinct original volume equations.")
        identities.add(identity)
        added_body_dofs = candidate.record["body_dofs"] - baseline.record["body_dofs"]
        cases[region] = {**_compare_prediction(parent_work, selection, edge, central_edge,
                                               added_body_dofs=added_body_dofs),
                         "native_response": taper._case_facts(candidate),
                         "mode_matches_and_original_increment_limits": {key: edge[key] for key in (
                             "signed_mode_matches", "section_relative_maximum_entry_changes",
                             "maximum_absolute_total_drive_power_change", "total_drive_power_increment_passed",
                             "maximum_absolute_complex_amplitude_change", "complex_amplitude_increment_passed")}}
    return {**native.AUTHOR, **native.CONVERGENCE, "schema_version": 1,
            "status": "compared_saved_goal_patch_controls", "frequency_hz": taper.FREQUENCY_GHZ * 1e9,
            "physical_thickness_nm": 100, "incident_side": "left",
            "reference_planes_x_m": prior["reference_planes_x_m"],
            "original_numerical_limits": prior["original_numerical_limits"],
            "sign_and_ranking_resolution_percentage_points": SIGNIFICANT_PP,
            f"ranking_resolution_percentage_points{BODY_DOF_SUFFIX}": _per_dof_ranking_resolution(cases),
            "coarse": taper._case_facts(baseline), "full_central": taper._case_facts(center),
            "full_central_signed_mode_matches": central_edge["signed_mode_matches"],
            "full_central_work_incident_roots": work_roots, "cases": cases, "drive_3_5_checks": _drive_checks(cases),
            "scope": "The predictor sums only terms attributed to the coarse parents actually replaced by each pair mesh, "
                     "using fields from the original coarse and full-central finite response. Full-bin work includes the "
                     "unreplaced part of that endpoint prediction. Each actual pair response is independently measured "
                     "against the coarse mesh. Prediction errors can reflect changed fields and coupling. The pair-to-center "
                     "increment follows that specific mesh path. Transmission change per 1000 added body DOFs compares "
                     "these two finite mesh steps. Timing and continuum error require independent evidence. Spatial and "
                     "frequency convergence remain unresolved.",
            "provenance": {"parent_work": work_pin, "published_donor_comparison": native._link(taper.PRIOR, ROOT),
                           "acceptance_reader": reader_pin, "original_mesh_donors": donors,
                           "sources": [native._link(Path(path).resolve(), ROOT)
                                       for path in (__file__, taper.__file__, runner.__file__, selector.__file__)]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=taper.STUDY)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    try:
        require(not output.is_relative_to(args.study_root.resolve())
                and output not in (args.work.resolve(), taper.PRIOR), "Write to a distinct file outside the saved run archive.")
        result = replay(args.study_root, args.work)
        with NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        try:
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    except (ArithmeticError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(1, f"Goal patch comparison failed. {exc}\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
