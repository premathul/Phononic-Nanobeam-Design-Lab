"""Replay the through-thickness control for the axial frequency pair. Athul Prem | _AP_pnc."""

from pathlib import Path

from tools import compare_axial_device_power_AP_pnc as axial
from tools import compare_axial_frequency_pair_AP_pnc as pair
from tools import map_scattering_AP_pnc as native
from tools.compare_scattering_AP_pnc import require


ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "verification/axial_device_three_layer_pairs_1GHz_AP_pnc.json"
FREQUENCIES = {"995MHz": pair.LOWER_HZ / 1e9, "1000MHz": pair.UPPER_HZ / 1e9}
CONTRAST = "thin_minus_thick_R_percentage_points"
CHANGE = "995_minus_1000MHz"


def _read_endpoint(directory, flux_path, frequency, mesh, thickness, planes, reader, replay_sources):
    accepted, configuration, holes = reader(directory, flux_path, frequency)
    replay = native._read(flux_path)
    require(configuration["mesh"] == mesh and accepted["physical_thickness_nm"] == thickness
            and accepted["reference_planes_x_m"] == planes
            and [replay[key] for key in ("replay_source_sha256", "regional_replay_source_sha256")] == replay_sources,
            f"Discretization, end planes or independent flux producers differ for {directory}.")
    record, selected = axial._native_point(directory, frequency)
    point = axial._read_case(directory, record, selected)
    index, reason = axial._axial_choice(point.channels)
    require(reason is None, f"{directory}: {reason}")
    mode = point.channels[index]
    replayed = accepted["native_drives"][index]
    require((mode["root"], mode["k_rad_m"]) == (replayed["native_root_id"], replayed["k_rad_m"])
            and all(abs(mode[key] - replayed[power]) <= 1e-12
                    for key, power in (("T", "transmission"), ("R", "reflection"))),
            f"The matched axial drive differs from the independently replayed drive for {directory}.")
    return {
        **mode, "body_dofs": accepted["body_dofs"], "port_dofs": accepted["port_dofs"],
        "all_incident_drives_checked": len(accepted["native_drives"]),
        "maximum_native_checks": accepted["maximum_numerical_checks"],
        "independent_flux_checks": accepted["independent_flux_checks"],
        "archive": {key: accepted["archive"][key] for key in pair.POINT_PINS},
    }, (point, index), selected[2], holes


def _match(reference, candidate):
    first, i = reference
    second, j = candidate
    forward, forward_competing = axial._match_evidence(axial._overlap(first, second), i, j)
    reverse, reverse_competing = axial._match_evidence(axial._overlap(second, first), j, i)
    require(first.settings == second.settings and min(forward, reverse) >= axial.MIN_OVERLAP
            and forward > forward_competing + axial.K_GAP_TOL and reverse > reverse_competing + axial.K_GAP_TOL,
            "The saved layer-control axial drives lack a unique reciprocal signed match.")
    return {"minimum_reciprocal_overlap": min(forward, reverse),
            "maximum_competing_overlap": max(forward_competing, reverse_competing)}


def _verify_seed(reference_dir, candidate_dir, donor, seeded, reference, candidate, seeder):
    shared = donor["shared"]
    require(seeded["inputs"] == {**donor["inputs"], "frequencies_GHz": [FREQUENCIES["995MHz"]]}
            and donor["inputs"]["frequencies_GHz"] == [FREQUENCIES["1000MHz"]]
            and seeded["shared"] == shared
            and seeded["preparation_reuse"] == {
                "donor_checkpoint_sha256": reference["archive"]["checkpoint"]["sha256"],
                "donor_shared_sha256": native.workflow.digest(shared),
                "donor_source_git_commit": donor["source_git_commit"],
                "seeder_source_sha256": seeder["sha256"],
            }
            and all(candidate[key] == reference[key] for key in ("body_dofs", "port_dofs")),
            f"Seeded preparation or numerical inputs differ from the pinned 1 GHz donor: {candidate_dir}.")
    for name in native.SHARED_FILES:
        link = shared["artifacts"][name]
        native._verified(link, reference_dir)
        native._verified(link, candidate_dir)
    return {name: shared["artifacts"][name]["sha256"] for name in native.SHARED_FILES}


def _sample_case(study_root, pilot, anchor, layer, role, mesh, reader, replay_sources, seeder):
    thickness = pair.ROLES[role][0]
    expected = anchor["maps"][f"{role}_z{layer}"]
    reference_dir = (ROOT / expected["map_directory"]).resolve()
    flux_parent = reference_dir.parent / "local_checks" if reference_dir.parent.name == "regional_mesh_AP_pnc" else reference_dir.parent
    reference_flux = flux_parent / f"{reference_dir.name.removesuffix('_1GHz')}_flux_AP_pnc.json"
    layer_id = "rim32" if layer == 2 else f"z{layer}"
    candidate_dir = study_root / f"{role}{thickness}_central400_{layer_id}_995MHz"
    candidate_flux = candidate_dir.with_name(f"{candidate_dir.name}_flux_AP_pnc.json")
    planes = anchor["reference_planes_x_m"]
    reference, mode_ref, donor, holes_ref = _read_endpoint(
        reference_dir, reference_flux, FREQUENCIES["1000MHz"], mesh, thickness, planes, reader, replay_sources)
    candidate, mode_new, seeded, holes_new = _read_endpoint(
        candidate_dir, candidate_flux, FREQUENCIES["995MHz"], mesh, thickness, planes, reader, replay_sources)
    require(all(reference["archive"][key]["sha256"] == expected[f"{key}_sha256"] for key in ("map", "checkpoint"))
            and pair._same_mode(reference, expected["native_axial_drive"])
            and holes_ref == holes_new and holes_new["source_sha256"] == anchor["hole_source_sha256"],
            f"The 1 GHz donor or resolved hole pattern differs from the published layer control: {reference_dir}.")
    if layer == 2:
        frozen = pilot["rim32_central_refinement"]["samples"][0]
        lower, upper = (frozen[f"axial_R_{frequency}"][role] for frequency in ("995MHz", "1000MHz"))
        require(frozen["central_target_nm"] == 400
                and all(lower[key] == candidate["archive"][archive] for key, archive in (
                    ("native_map", "map"), ("native_point_summary", "point_summary"),
                    ("independent_flux_replay", "independent_flux_replay"),
                ))
                and all(abs(saved["R"] - actual["R"]) <= 1e-12 and all(saved[key] == actual[key] for key in ("drive", "root"))
                        for saved, actual in ((lower, candidate), (upper, reference))),
                f"The two-layer responses differ from the frozen 400 nm axial pilot for {role}.")
    preparation = _verify_seed(reference_dir, candidate_dir, donor, seeded, reference, candidate, seeder)
    return {
        "995MHz": candidate, "1000MHz": reference, "identical_preparation_sha256": preparation,
        "within_thickness_endpoint_match": _match(mode_ref, mode_new),
    }, {"995MHz": mode_new, "1000MHz": mode_ref}


def compare_layer_frequency(study_root, pilot, reader, replay_sources):
    """Return the fixed-400 nm layer series for the paired fourth-mesh record."""
    study_root = Path(study_root).resolve()
    anchor = native._read(ANCHOR)
    require(anchor["status"] == "completed_finite_paired_comparison"
            and anchor["central_target_nm"] == 400 and anchor["lead_target_nm"] == 150
            and anchor["reference_planes_x_m"] == pilot["reference_planes_x_m"]
            and anchor["hole_count"] == pilot["hole_count"] == 26
            and anchor["original_numerical_limits"] == pilot["original_numerical_limits"],
            "Expected the published 400/150 nm layer control with the pilot's original geometry and numerical limits.")
    seeder = native._link(ROOT / "tools/seed_frequency_map_AP_pnc.py", ROOT)
    samples, modes = [], {}
    base = {"kind": "layered", "element_order": 2, "mesh_size_nm": 400, "port_size_nm": 250,
            "circle_resolution": 32, "lead_mesh_size_nm": 150, "patterned_x_interval_nm": [1000, 15650]}
    for layer in (1, 2, 3):
        cases = {}
        for role in pair.ROLES:
            cases[role], modes[role, layer] = _sample_case(
                study_root, pilot, anchor, layer, role, {**base, "thickness_layers": layer}, reader, replay_sources, seeder)
        contrasts = {label: 100 * (cases["thin"][label]["R"] - cases["thick"][label]["R"]) for label in FREQUENCIES}
        contrasts[CHANGE] = contrasts["995MHz"] - contrasts["1000MHz"]
        samples.append({"quadratic_layers": layer, **cases, CONTRAST: contrasts,
                        "cross_thickness_matches": {label: _match(modes["thick", layer][label], modes["thin", layer][label])
                                                   for label in FREQUENCIES}})
    return {
        "status": "completed_finite_through_thickness_comparison",
        "central_target_nm": 400, "lead_target_nm": 150, "hole_wall_segments": 32,
        "samples": samples,
        "signed_endpoint_change_increments_percentage_points": {
            f"{first}_to_{second}": samples[second - 1][CONTRAST][CHANGE] - samples[first - 1][CONTRAST][CHANGE]
            for first, second in ((1, 2), (2, 3))
        },
        "successive_layer_matches_at_995MHz": {
            f"{first}_to_{second}": {role: _match(modes[role, first]["995MHz"], modes[role, second]["995MHz"])
                                    for role in pair.ROLES}
            for first, second in ((1, 2), (2, 3))
        },
        "provenance": {"published_1GHz_layer_control": native._link(ANCHOR, ROOT),
                       "seeder": seeder, "comparison_source": native._link(Path(__file__).resolve(), ROOT)},
        "scope": "The layer step changes the body and the compatible port discretizations together at fixed in-plane settings.",
    }
