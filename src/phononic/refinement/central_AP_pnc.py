"""Nested planar refinement for regional beams. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

from collections import defaultdict
from heapq import heappop, heappush
from math import dist, isfinite
from numbers import Integral, Real

import numpy as np


def _edges(triangle):
    a, b, c = sorted(triangle)
    return (a, b), (a, c), (b, c)


def refine_footprint(xy, triangles, interval, max_edge_m, *, max_triangles=None):
    """Bisect a regional footprint in metres, preserving the donor vertex prefix.

    The input is the native axis-aligned beam footprint: (N, 2) coordinates,
    (M, 3) vertex indices and two exact x cut coordinates. Lead triangles,
    cut edges and hole-rim edges stay intact. Straight central outer sides may
    be subdivided without moving the polygon boundary.

    The ceiling on each eligible edge is the largest of ``max_edge_m`` and
    the two cut-trace limits. For each cut, its limit is the longest edge
    on the trace minus half the x distance of the nearer endpoint to that
    cut. This fixed grading permits long edges near the protected cuts.
    The requested target must be at least the longest frozen rim edge.

    Longest eligible edges are split first, breaking ties by vertex indices.
    Lowering the target continues the same split sequence. Each midpoint is
    shared by both incident triangles. New triangles have sorted indices for
    the layered extrusion, and untouched triangles keep their original slots.
    """
    xy = np.asarray(xy, dtype=float)
    triangles = np.asarray(triangles)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError("Footprint coordinates must be a finite (N, 2) array")
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or triangles.dtype.kind not in "iu" or triangles.min() < 0 or triangles.max() >= len(xy)):
        raise ValueError("Footprint triangles must be an (M, 3) array of vertex indices")
    start, stop = interval
    if not xy[:, 0].min() < start < stop < xy[:, 0].max():
        raise ValueError("Central cuts must be strictly ordered inside the footprint")
    if (isinstance(max_edge_m, bool) or not isinstance(max_edge_m, Real)
            or not isfinite(max_edge_m) or max_edge_m <= 0):
        raise ValueError("The central edge target must be positive and finite")
    if max_triangles is not None and (isinstance(max_triangles, bool)
            or not isinstance(max_triangles, Integral) or max_triangles < len(triangles)):
        raise ValueError("The planar triangle limit must admit the donor footprint")

    points = [tuple(point) for point in xy.tolist()]
    faces = [tuple(face) for face in triangles.tolist()]
    x = xy[triangles, 0]
    xmin, xmax = x.min(axis=1), x.max(axis=1)
    if np.any(((xmin < start) & (xmax > start)) | ((xmin < stop) & (xmax > stop))):
        raise ValueError("A footprint triangle crosses a central cut")
    incident = defaultdict(set)
    for index in np.flatnonzero((xmin >= start) & (xmax <= stop)):
        for edge in _edges(faces[index]):
            incident[edge].add(int(index))

    outer_y = (float(xy[:, 1].min()), float(xy[:, 1].max()))
    cuts = {start: 0., stop: 0.}
    rim_max = 0.
    frozen = set()
    for edge, neighbors in incident.items():
        if len(neighbors) > 2:
            raise ValueError("A central edge has more than two incident triangles")
        if len(neighbors) == 2:
            continue
        a, b = (points[vertex] for vertex in edge)
        if a[0] == b[0] and a[0] in cuts:
            cuts[a[0]] = max(cuts[a[0]], dist(a, b))
            frozen.add(edge)
        elif not (a[1] == b[1] and a[1] in outer_y):
            rim_max = max(rim_max, dist(a, b))
            frozen.add(edge)
    if not all(cuts.values()):
        raise ValueError("The central footprint must have both fixed cut traces")
    if max_edge_m < rim_max:
        raise ValueError(f"The central edge target is below the longest frozen hole-rim edge ({rim_max:g} m)")

    queue = []

    def queue_edge(edge):
        if edge in frozen:
            return
        first, second = (points[vertex] for vertex in edge)
        left_limit = cuts[start] - (min(first[0], second[0]) - start) / 2
        right_limit = cuts[stop] - (stop - max(first[0], second[0])) / 2
        length = dist(first, second)
        if length > max(max_edge_m, left_limit, right_limit):
            heappush(queue, (-length, *edge))

    for edge in incident:
        queue_edge(edge)
    while queue:
        _, a, b = heappop(queue)
        neighbors = sorted(incident[(a, b)])
        if max_triangles is not None and len(faces) + len(neighbors) > max_triangles:
            raise ValueError(f"Central refinement exceeds the planar triangle limit of {max_triangles}")

        midpoint = len(points)
        point = tuple(p + (q - p) / 2 for p, q in zip(points[a], points[b]))
        if point in (points[a], points[b]):
            raise ValueError("The central edge target is below the footprint coordinate precision")
        points.append(point)
        new_edges = set()
        for index in neighbors:
            original = faces[index]
            opposite, = (vertex for vertex in original if vertex not in (a, b))
            for edge in _edges(original):
                incident[edge].remove(index)
                if not incident[edge]:
                    del incident[edge]

            sibling = len(faces)
            faces[index] = (*sorted((a, opposite)), midpoint)
            faces.append((*sorted((b, opposite)), midpoint))
            for child in (index, sibling):
                for edge in _edges(faces[child]):
                    incident[edge].add(child)
                    if edge[1] == midpoint:
                        new_edges.add(edge)
        for edge in new_edges:
            queue_edge(edge)

    return np.asarray(points, dtype=xy.dtype), np.asarray(faces, dtype=triangles.dtype)
