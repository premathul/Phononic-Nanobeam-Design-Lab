"""Select two conforming refinements from exact saved donor cells. Athul Prem | _AP_pnc."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skfem import MeshTet

from tools import central_patch_refinement_AP_pnc as patch


REGIONS = {"cavity_13_14": (13, 14), "outer_taper_09_18": (9, 18)}


def _filtered_donor(coarse, fine, coarse_columns, fine_columns, levels, keep, add):
    """Limit the fine donor to selected replacements in native layer order."""
    used_columns = np.unique(fine.t[:, add] % fine_columns)
    columns = np.r_[np.arange(coarse_columns), used_columns[used_columns >= coarse_columns]]
    point_ids = (np.arange(levels)[:, None] * fine_columns + columns).ravel()
    fine_to_filtered = np.full(fine.nvertices, -1)
    fine_to_filtered[point_ids] = np.arange(len(point_ids))
    coarse_to_filtered = (np.arange(levels)[:, None] * len(columns) + np.arange(coarse_columns)).ravel()
    tetrahedra = np.column_stack((coarse_to_filtered[coarse.t[:, keep]], fine_to_filtered[fine.t[:, add]]))
    return MeshTet(np.ascontiguousarray(fine.p[:, point_ids]), np.ascontiguousarray(tetrahedra))


def make_goal_patch_control(
    coarse: MeshTet,
    fine: MeshTet,
    cuts_m: tuple[float, float],
    hole_centers_m,
    *,
    region: str,
) -> tuple[MeshTet, dict]:
    """Build a goal patch in the original central partition of the coarse mesh.

    A coarse parent is selected by its planar centroid's nearest axial hole
    centre. Components joined by split coarse edges stay coarse if they
    touch an unselected parent. The resulting mesh uses exact original
    retained coarse cells and their selected fine donor replacements.
    """
    if region not in REGIONS:
        raise ValueError(f"Choose one of the goal regions: {', '.join(REGIONS)}.")
    centers = np.asarray(hole_centers_m, dtype=float)
    if centers.shape != (26,) or not np.isfinite(centers).all() or not np.all(np.diff(centers) > 0):
        raise ValueError("Expected the 26 ordered original axial hole centres.")
    bin_edges = (centers[1:] + centers[:-1]) / 2
    z, xy, planar, coarse_owner = patch._footprint(coarse)
    fine_z, fine_xy, fine_planar, fine_owner = patch._footprint(fine)
    lower, upper = cuts_m
    if not xy[:, 0].min() < lower < upper < xy[:, 0].max():
        raise ValueError("Place both ordered original central cuts strictly inside the beam.")
    if not np.array_equal(z, fine_z) or not np.array_equal(xy, fine_xy[:len(xy)]):
        raise ValueError("The original footprints must share coarse vertex coordinates, order and z planes.")

    triangles = xy[planar]
    fine_parents = patch._descendant_parents(triangles, fine_xy[fine_planar])
    split = np.bincount(fine_parents, minlength=len(planar)) > 1
    x = triangles[:, :, 0]
    inside = (x.min(axis=1) >= lower) & (x.max(axis=1) <= upper)
    central = patch._compatible_patch(planar, fine_planar, inside)
    # Sort corner x coordinates before averaging so floating-point bin ownership
    # matches the coordinate-sorted columns in the original work comparison.
    parent_holes = np.searchsorted(bin_edges, np.sort(x, axis=1).mean(axis=1)) + 1
    in_bins = np.isin(parent_holes, REGIONS[region])
    selected = central & in_bins
    compatible = patch._compatible_patch(planar, fine_planar, selected)
    replaced = compatible & split
    if not replaced.any():
        raise ValueError(f"No nested refinement lies inside the conforming {region} patch.")
    keep = ~replaced[coarse_owner]
    add = replaced[fine_parents[fine_owner]]

    filtered = _filtered_donor(coarse, fine, len(xy), len(fine_xy), len(z), keep, add)
    control, shared = patch.make_central_patch_control(coarse, filtered, cuts_m)
    n_retained = int(keep.sum())
    expected_coarse = coarse.p[:, coarse.t[:, keep]]
    expected_fine = fine.p[:, fine.t[:, add]]
    if (control.nelements != n_retained + add.sum()
            or not np.array_equal(control.p[:, control.t[:, :n_retained]], expected_coarse)
            or not np.array_equal(control.p[:, control.t[:, n_retained:]], expected_fine)):
        raise ValueError("The goal patch differs from the original retained coarse or inserted fine 3D cells.")
    actual_frontier = patch._frontier(control, np.arange(control.nelements) >= n_retained)
    coarse_frontier = patch._frontier(coarse, ~keep)
    if not actual_frontier.size or patch._facet_keys(actual_frontier) != patch._facet_keys(coarse_frontier):
        raise ValueError("The goal patch changed its coarse interface vertices, edges or faces.")

    native = patch.native
    diagnostics = {
        **{key: value for key, value in shared.items() if key != "coarse_parent_partition"},
        "method": "nested_donor_refinement_in_selected_axial_hole_bins",
        "region": region,
        "control_source_sha256": native._sha(Path(__file__)),
        "shared_geometry_source_sha256": shared["control_source_sha256"],
        "fine_coordinates_and_connectivity_sha256": native._array_digest(fine.p, fine.t),
        "fine_donor_planar_triangles": len(fine_planar),
        "filtered_donor_coordinates_and_connectivity_sha256": shared["fine_coordinates_and_connectivity_sha256"],
        "wholly_inside_coarse_triangles_frozen_by_conformity": int((inside & ~central).sum()),
        "retained_coarse_tetrahedra_geometry_sha256": native._array_digest(expected_coarse),
        "inserted_fine_donor_tetrahedra_geometry_sha256": native._array_digest(expected_fine),
        "unchanged_patch_frontier_facets": actual_frontier.shape[1],
        "unchanged_patch_frontier_p2_dofs": patch._p2_dofs(actual_frontier),
        "goal_parent_partition": {
            "index_order": "Zero-based original native bottom-footprint triangle order.",
            "hole_bins": list(REGIONS[region]),
            "hole_bin_edges_x_m": [[float(bin_edges[hole - 2]), float(bin_edges[hole - 1])] for hole in REGIONS[region]],
            "central_parent_indices": np.flatnonzero(central).tolist(),
            "selected_parent_indices": np.flatnonzero(selected).tolist(),
            "selected_refinable_parent_indices": np.flatnonzero(selected & split).tolist(),
            "selected_parent_indices_frozen_by_conformity": np.flatnonzero(selected & ~compatible).tolist(),
            "replaced_parent_indices": np.flatnonzero(replaced).tolist(),
            "bins_parent_count_outside_original_central_mask": int((in_bins & ~central).sum()),
            "parent_counts_by_hole": {
                str(hole): {
                    "selected": int((selected & (parent_holes == hole)).sum()),
                    "selected_refinable": int((selected & split & (parent_holes == hole)).sum()),
                    "replaced": int((replaced & (parent_holes == hole)).sum()),
                }
                for hole in REGIONS[region]
            },
        },
    }
    return control, diagnostics
