"""Mesh segmentation post-processing and boundary graph cut.

The pipeline keeps real shared-edge adjacency separate from augmented cross-shell
adjacency. Real adjacency is used for connected components, area statistics, and
graph-cut smoothness. Augmented adjacency is used only for label propagation.
Semantic labels remain aligned with generated ``<|SEG|>`` tokens; optional instance
splits map back to semantic classes through ``instance_to_class``.
"""

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

try:
    import igraph as _ig
except ImportError:
    _ig = None

_INF = float("inf")






def build_adjacency(combined_mesh):
    """Return (real_nbr, real_adjacency, shell_ids, num_shells).

    real_nbr: list of shared-edge neighbors for every face
    real_adjacency: [E, 2] ndarray
    shell_ids: [F] face-adjacency component ID for every face
    """
    num_faces = len(combined_mesh.faces)
    adjacency = np.asarray(combined_mesh.face_adjacency, dtype=np.int64)
    if adjacency.size == 0:
        adjacency = np.zeros((0, 2), dtype=np.int64)

    real_nbr = [[] for _ in range(num_faces)]
    for a, b in adjacency:
        real_nbr[a].append(int(b))
        real_nbr[b].append(int(a))

    if len(adjacency) > 0:
        graph = sp.coo_matrix(
            (np.ones(len(adjacency)), (adjacency[:, 0], adjacency[:, 1])),
            shape=(num_faces, num_faces),
        )
        num_shells, shell_ids = connected_components(graph, directed=False)
    else:
        num_shells, shell_ids = num_faces, np.arange(num_faces)

    return real_nbr, adjacency, shell_ids, num_shells


def augment_across_shells(real_nbr, shell_ids, num_shells, combined_mesh):
    """Add virtual edges between shells so labels cross topology gaps.

    Shells are connected by centroid proximity instead of face-index order.

    Returns a new adjacency list without modifying ``real_nbr``.
    """
    aug_nbr = [list(v) for v in real_nbr]
    if num_shells <= 1:
        return aug_nbr

    centers = combined_mesh.triangles_center
    areas = combined_mesh.area_faces


    shell_centroid = np.zeros((num_shells, 3))
    shell_repr = np.zeros(num_shells, dtype=np.int64)
    order = np.argsort(shell_ids, kind="stable")
    sorted_shell = shell_ids[order]
    boundaries = np.searchsorted(sorted_shell, np.arange(num_shells + 1))
    for s in range(num_shells):
        idx = order[boundaries[s]:boundaries[s + 1]]
        if len(idx) == 0:
            continue
        w = areas[idx]
        total = w.sum()
        shell_centroid[s] = (centers[idx] * w[:, None]).sum(0) / total if total > 0 else centers[idx].mean(0)
        shell_repr[s] = idx[np.argmax(w)]


    from scipy.spatial import cKDTree

    tree = cKDTree(shell_centroid)
    k = min(3, num_shells)
    _, nn = tree.query(shell_centroid, k=k)
    nn = np.atleast_2d(nn)
    for s in range(num_shells):
        for cand in nn[s]:
            if cand == s:
                continue
            f1, f2 = int(shell_repr[s]), int(shell_repr[cand])
            if f2 not in aug_nbr[f1]:
                aug_nbr[f1].append(f2)
            if f1 not in aug_nbr[f2]:
                aug_nbr[f2].append(f1)
            break

    return aug_nbr






def fill_by_voting(face_labels, nbr, iterations=16):
    """Fill -1 labels through iterative neighborhood majority voting.

    Updates are applied synchronously so traversal order does not affect propagation.
    """
    labels = np.asarray(face_labels).copy()
    for _ in range(iterations):
        unlabeled = np.where(labels == -1)[0]
        if len(unlabeled) == 0:
            break
        changes = {}
        for face in unlabeled:
            counter = Counter()
            for adj in nbr[face]:
                if labels[adj] != -1:
                    counter[int(labels[adj])] += 1
            if counter:

                best = max(counter.items(), key=lambda kv: (kv[1], -kv[0]))
                changes[face] = best[0]
        if not changes:
            break
        for face, label in changes.items():
            labels[face] = label
    return labels


def fill_remaining_by_kdtree(face_labels, combined_mesh):
    """Assign each remaining -1 face the label of its nearest labeled face."""
    labels = np.asarray(face_labels).copy()
    remaining = np.where(labels == -1)[0]
    if len(remaining) == 0:
        return labels
    labeled = np.where(labels != -1)[0]
    if len(labeled) == 0:
        return labels
    from scipy.spatial import cKDTree

    centers = combined_mesh.triangles_center
    _, nn = cKDTree(centers[labeled]).query(centers[remaining])
    labels[remaining] = labels[labeled[nn]]
    return labels






def label_connected_regions(face_labels, real_adjacency, num_faces):
    """Return region IDs and count for same-label shared-edge components."""
    if len(real_adjacency) == 0:
        return np.arange(num_faces), num_faces
    same = face_labels[real_adjacency[:, 0]] == face_labels[real_adjacency[:, 1]]
    edges = real_adjacency[same]
    graph = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(num_faces, num_faces),
    )
    num_regions, region_ids = connected_components(graph, directed=False)
    return region_ids, num_regions


def drop_small_regions(face_labels, region_ids, num_regions, shell_ids,
                       combined_mesh, rel_area_threshold=0.001):
    """Remove small regions using area relative to the containing shell.

    Relative area protects small but valid shells such as buttons or eyes. Removed
    faces are set to -1 and refilled by the subsequent voting stage.
    """
    labels = np.asarray(face_labels).copy()
    areas = combined_mesh.area_faces

    region_area = np.bincount(region_ids, weights=areas, minlength=num_regions)

    region_shell = np.zeros(num_regions, dtype=np.int64)
    region_shell[region_ids] = shell_ids
    shell_area = np.bincount(shell_ids, weights=areas)

    denom = shell_area[region_shell]
    ratio = region_area / (denom + 1e-7)
    drop = np.where(ratio < rel_area_threshold)[0]
    if len(drop) == 0:
        return labels, 0

    drop_mask = np.isin(region_ids, drop)

    for shell in np.unique(shell_ids[drop_mask]):
        in_shell = shell_ids == shell
        if drop_mask[in_shell].all():
            regions_here = np.unique(region_ids[in_shell])
            keep = regions_here[np.argmax(region_area[regions_here])]
            drop_mask[in_shell & (region_ids == keep)] = False

    labels[drop_mask] = -1
    return labels, int(drop_mask.sum())


def merge_area_tail_regions(face_labels, real_adjacency, combined_mesh,
                            cumulative_threshold=0.95, max_part_area=0.01):
    """Merge tail regions beyond the cumulative-area threshold.

    Small regions beyond ``cumulative_threshold`` are merged into their largest
    adjacent non-tail region.
    """
    labels = np.asarray(face_labels).copy()
    num_faces = len(labels)
    region_ids, num_regions = label_connected_regions(labels, real_adjacency, num_faces)
    if num_regions <= 1:
        return labels, 0

    areas = combined_mesh.area_faces
    region_area = np.bincount(region_ids, weights=areas, minlength=num_regions)
    total = region_area.sum()
    if total <= 0:
        return labels, 0
    frac = region_area / total

    order = np.argsort(-frac)
    cumulative = np.cumsum(frac[order])
    rank = np.empty(num_regions, dtype=np.int64)
    rank[order] = np.arange(num_regions)
    is_tail = cumulative[rank] > cumulative_threshold


    if len(real_adjacency) == 0:
        return labels, 0
    ra, rb = region_ids[real_adjacency[:, 0]], region_ids[real_adjacency[:, 1]]
    cross = ra != rb
    ra, rb = ra[cross], rb[cross]

    neighbors = {}
    for x, y in zip(ra, rb):
        neighbors.setdefault(int(x), set()).add(int(y))
        neighbors.setdefault(int(y), set()).add(int(x))

    num_merged = 0
    for region in order:
        if not is_tail[region] or frac[region] >= max_part_area:
            continue
        cands = [n for n in neighbors.get(int(region), ()) if not is_tail[n]]
        if not cands:
            continue
        target = max(cands, key=lambda n: region_area[n])
        target_face = np.where(region_ids == target)[0]
        if len(target_face) == 0:
            continue
        labels[region_ids == region] = labels[target_face[0]]
        num_merged += 1

    return labels, num_merged






def split_instances(face_labels, real_adjacency, num_faces):
    """Split disconnected regions of one semantic class into instances.

    Returns:
        instance_labels: [F] instance IDs in 0..K-1
        instance_to_class: [K] original semantic class index for each instance
    """
    region_ids, num_regions = label_connected_regions(face_labels, real_adjacency, num_faces)
    instance_to_class = np.zeros(num_regions, dtype=np.int64)
    for region in range(num_regions):
        faces = np.where(region_ids == region)[0]
        if len(faces) > 0:
            instance_to_class[region] = face_labels[faces[0]]
    return region_ids, instance_to_class






def postprocess_face_labels(
    face_labels,
    combined_mesh,
    stitch_shells=True,
    vote_iterations=16,
    drop_small=True,
    rel_area_threshold=0.001,
    merge_area_tail=True,
    cumulative_threshold=0.95,
    split=False,
    verbose=False,
):
    """Run the complete mesh post-processing pipeline before graph cut.

    Args:
        face_labels: [F] integers; -1 denotes an unlabeled face
        combined_mesh: trimesh.Trimesh
        stitch_shells: add cross-shell virtual edges for label filling
        vote_iterations: number of iterative voting rounds
        drop_small: remove regions below the shell-relative area threshold
        merge_area_tail: merge regions in the cumulative-area tail
        split: split disconnected instances

    Returns:
        Dictionary containing face_labels, optional instance mappings, and stats.
    """
    labels = np.asarray(face_labels).copy().astype(np.int64)
    num_faces = len(labels)
    stats = {"num_unlabeled_input": int((labels == -1).sum())}

    real_nbr, real_adjacency, shell_ids, num_shells = build_adjacency(combined_mesh)
    stats["num_shells"] = int(num_shells)

    fill_nbr = (augment_across_shells(real_nbr, shell_ids, num_shells, combined_mesh)
                if stitch_shells else real_nbr)

    labels = fill_by_voting(labels, fill_nbr, iterations=vote_iterations)
    stats["num_unlabeled_after_vote"] = int((labels == -1).sum())
    labels = fill_remaining_by_kdtree(labels, combined_mesh)

    if drop_small:
        region_ids, num_regions = label_connected_regions(labels, real_adjacency, num_faces)
        stats["num_regions_before_drop"] = int(num_regions)
        labels, num_dropped = drop_small_regions(
            labels, region_ids, num_regions, shell_ids, combined_mesh,
            rel_area_threshold=rel_area_threshold,
        )
        stats["num_faces_dropped"] = num_dropped
        if num_dropped > 0:
            labels = fill_by_voting(labels, fill_nbr, iterations=vote_iterations)
            labels = fill_remaining_by_kdtree(labels, combined_mesh)

    if merge_area_tail:
        labels, num_merged = merge_area_tail_regions(
            labels, real_adjacency, combined_mesh,
            cumulative_threshold=cumulative_threshold,
        )
        stats["num_regions_merged"] = num_merged

    region_ids, num_regions = label_connected_regions(labels, real_adjacency, num_faces)
    stats["num_regions_final"] = int(num_regions)

    result = {"face_labels": labels, "stats": stats}
    if split:
        instance_labels, instance_to_class = split_instances(labels, real_adjacency, num_faces)
        result["instance_labels"] = instance_labels
        result["instance_to_class"] = instance_to_class
        stats["num_instances"] = int(len(instance_to_class))

    if verbose:
        print("  [PostProc] " + "  ".join(f"{k}={v}" for k, v in stats.items()))

    return result


def compute_data_costs(face_logits_sum, face_point_count):
    """Compute dense unary costs ``[F,C]`` from accumulated logits.

    This compatibility path supports the legacy inference API. Callers with explicit
    unary costs should use :class:`UnaryCostProvider` to avoid an ``F x C`` matrix.
    """
    F, C = face_logits_sum.shape
    sampled = face_point_count > 0
    probs = np.full((F, C), 1.0 / C)
    if sampled.any():
        avg = face_logits_sum[sampled] / face_point_count[sampled, None]
        avg = avg - avg.max(axis=1, keepdims=True)
        e = np.exp(avg)
        probs[sampled] = e / e.sum(axis=1, keepdims=True)


    return -np.log(np.minimum(probs + 1e-10, 1.0))


def compute_smooth_costs(mesh, _lambda=1.0, smooth_eps=1e-10,
                         smooth_mode="log"):
    """Compute pairwise costs ``[E]`` over shared mesh edges."""
    angles = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
    if smooth_mode == "concave":
        cost_smooth = (1.0 - angles / np.pi)
    else:
        cost_smooth = -np.log(np.minimum(angles / np.pi + smooth_eps, 1.0))
        if smooth_mode == "bounded" and len(cost_smooth):
            finite = cost_smooth[np.isfinite(cost_smooth)]
            if len(finite):
                hi = np.percentile(finite, 99.0)
                cost_smooth = np.minimum(cost_smooth, hi)
    return cost_smooth * float(_lambda)


def compute_costs(face_logits_sum, face_point_count, mesh, _lambda=1.0,
                  smooth_eps=1e-10, smooth_mode="log"):
    """Return ``(cost_data [F,C], cost_smooth [E])``.

    smooth_mode:
      "log": ``-log(angle/pi + eps)``
      "bounded": the log formulation clipped at p99
      "concave": ``1 - angle/pi``, bounded to [0, 1]
    """
    return (
        compute_data_costs(face_logits_sum, face_point_count),
        compute_smooth_costs(
            mesh, _lambda=_lambda, smooth_eps=smooth_eps,
            smooth_mode=smooth_mode,
        ),
    )


class UnaryCostProvider:
    """Unary-cost interface for alpha expansion.

    Each min-cut only needs costs for switching to alpha and retaining the current
    label. Subclasses implement ``pair_costs`` without constructing global costs.
    """

    def pair_costs(self, face_ids, alpha, keep_labels):
        raise NotImplementedError


class DenseUnaryCostProvider(UnaryCostProvider):
    """Wrap legacy ``[F,C]`` costs for API compatibility and regression tests."""

    def __init__(self, cost_data):
        self.cost_data = np.asarray(cost_data)

    def pair_costs(self, face_ids, alpha, keep_labels):
        face_ids = np.asarray(face_ids, dtype=np.int64)
        keep_labels = np.asarray(keep_labels, dtype=np.int64)
        rows = np.arange(len(face_ids), dtype=np.int64)
        local = self.cost_data[face_ids]
        return local[:, int(alpha)], local[rows, keep_labels]


class LogitsUnaryCostProvider(UnaryCostProvider):
    """Build legacy logit unary costs lazily after the complexity preflight."""

    def __init__(self, face_logits_sum, face_point_count):
        self.face_logits_sum = face_logits_sum
        self.face_point_count = face_point_count
        self._dense = None

    def _provider(self):
        if self._dense is None:
            self._dense = DenseUnaryCostProvider(
                compute_data_costs(self.face_logits_sum, self.face_point_count)
            )
        return self._dense

    def pair_costs(self, face_ids, alpha, keep_labels):
        return self._provider().pair_costs(face_ids, alpha, keep_labels)


class HardLabelUnaryCostProvider(UnaryCostProvider):
    """O(F) hard-label unary costs equivalent to the legacy one-hot path.

    ``seed_labels`` are fixed owners before graph cut. Retaining the seed label uses
    ``right_cost`` and switching uses ``wrong_cost``. Both follow the legacy numeric
    path to preserve per-face regression stability.
    """

    def __init__(self, seed_labels, num_labels, data_strength=1.0,
                 probability_eps=1e-10):
        self.seed_labels = np.asarray(seed_labels, dtype=np.int64).copy()
        self.num_labels = int(num_labels)
        if self.num_labels <= 0:
            raise ValueError("num_labels must be positive")
        strength = float(np.float32(data_strength))
        denom = 1.0 + (self.num_labels - 1) * np.exp(-strength)
        right_prob = 1.0 / denom
        wrong_prob = np.exp(-strength) / denom
        self.right_cost = float(-np.log(min(right_prob + probability_eps, 1.0)))
        self.wrong_cost = float(-np.log(min(wrong_prob + probability_eps, 1.0)))

    def pair_costs(self, face_ids, alpha, keep_labels):
        face_ids = np.asarray(face_ids, dtype=np.int64)
        keep_labels = np.asarray(keep_labels, dtype=np.int64)
        seed = self.seed_labels[face_ids]
        alpha_cost = np.where(
            seed == int(alpha), self.right_cost, self.wrong_cost
        )
        keep_cost = np.where(
            seed == keep_labels, self.right_cost, self.wrong_cost
        )
        return alpha_cost, keep_cost






def build_neighbor_index(mesh):
    """Build a reusable CSR neighbor index ``(indptr, indices)``.

    Boundary dilation then uses direct index lookup and touches only neighbors of the
    current frontier instead of rebuilding an F-by-F sparse matrix.
    """
    F = len(mesh.faces)
    fa = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(fa) == 0:
        return np.zeros(F + 1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    src = np.concatenate([fa[:, 0], fa[:, 1]])
    dst = np.concatenate([fa[:, 1], fa[:, 0]])
    order = np.argsort(src, kind="stable")
    indices = dst[order]
    counts = np.bincount(src, minlength=F)
    indptr = np.zeros(F + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, indices


def boundary_band(labels, mesh, rings=1, nbr_index=None):
    """Return an ``[F]`` mask of candidate faces within boundary rings.

    Dilation operates on the frontier, so cost scales with the band rather than the
    total face count.
    """
    F = len(labels)
    fa = np.asarray(mesh.face_adjacency, dtype=np.int64)
    band = np.zeros(F, dtype=bool)
    if len(fa) == 0:
        return band
    diff = labels[fa[:, 0]] != labels[fa[:, 1]]
    if not diff.any():
        return band
    seed = np.unique(fa[diff].reshape(-1))
    band[seed] = True
    if rings <= 0:
        return band

    if nbr_index is None:
        nbr_index = build_neighbor_index(mesh)
    indptr, indices = nbr_index

    frontier = seed
    for _ in range(int(rings)):
        if len(frontier) == 0:
            break

        lens = indptr[frontier + 1] - indptr[frontier]
        if lens.sum() == 0:
            break
        starts = np.repeat(indptr[frontier], lens)
        offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
        cand = indices[starts + offs]
        cand = cand[~band[cand]]
        if len(cand) == 0:
            break
        frontier = np.unique(cand)
        band[frontier] = True
    return band






def _solve_subproblem(args):
    """Run one alpha-expansion min-cut on a subset of faces.

    Neighbors outside the subset form fixed boundary conditions. Their smoothness
    costs become source or sink capacities. Edges and capacities are constructed in
    NumPy batches before being passed to igraph.

    Returns ``(face_indices, switch_to_alpha_mask)``.
    """
    (sub, label, partition_sub, cost_alpha_sub, cost_keep_sub,
     edges_local, edges_cost, bnd_to_alpha, bnd_to_sink) = args

    n = len(sub)

    is_label = partition_sub == label
    ar = np.arange(n)

    src_caps = cost_alpha_sub + bnd_to_alpha
    sink_caps = np.where(is_label, _INF, cost_keep_sub) + bnd_to_sink


    node = 2 + ar
    e_src = np.stack([np.zeros(n, dtype=np.int64), node], axis=1)
    e_snk = np.stack([node, np.ones(n, dtype=np.int64)], axis=1)
    edge_blocks = [e_src, e_snk]
    cap_blocks = [src_caps, sink_caps]

    n_aux = 0
    if len(edges_local):
        f1 = edges_local[:, 0]; f2 = edges_local[:, 1]
        p1 = partition_sub[f1]; p2 = partition_sub[f2]
        same = p1 == p2


        m = same & (p1 != label)
        if m.any():
            edge_blocks.append(np.stack([2 + f1[m], 2 + f2[m]], axis=1))
            cap_blocks.append(edges_cost[m])


        m2 = ~same
        n_aux = int(m2.sum())
        if n_aux:
            a1 = f1[m2]; a2 = f2[m2]; ac = edges_cost[m2]
            aux = 2 + n + np.arange(n_aux)
            edge_blocks.append(np.stack([aux, np.ones(n_aux, dtype=np.int64)], axis=1))
            cap_blocks.append(ac)
            k1 = partition_sub[a1] != label
            if k1.any():
                edge_blocks.append(np.stack([2 + a1[k1], aux[k1]], axis=1))
                cap_blocks.append(ac[k1])
            k2 = partition_sub[a2] != label
            if k2.any():
                edge_blocks.append(np.stack([aux[k2], 2 + a2[k2]], axis=1))
                cap_blocks.append(ac[k2])

    total = 2 + n + n_aux
    all_edges = np.concatenate(edge_blocks, axis=0)
    all_caps = np.concatenate(cap_blocks, axis=0)

    G = _ig.Graph(n=total, edges=all_edges.tolist(), directed=False)
    G.es["capacity"] = all_caps.tolist()
    r = G.st_mincut(source=0, target=1, capacity="capacity")
    in_sink_side = np.zeros(total, dtype=bool)
    in_sink_side[np.asarray(r.partition[1], dtype=np.int64)] = True
    return sub, in_sink_side[2:2 + n]






def _build_band_structure(labels, mesh, fa, cost_smooth, F, band_rings,
                          nbr_index=None):
    """Build the boundary band, components, and component edge groups once.

    Grouping avoids scanning the complete edge array for every label-component pair.
    """
    if band_rings is None:
        band = np.ones(F, dtype=bool)
    else:
        band = boundary_band(labels, mesh, rings=band_rings, nbr_index=nbr_index)
    if not band.any():
        return None

    in1 = band[fa[:, 0]]; in2 = band[fa[:, 1]]
    inner_mask = in1 & in2
    cross_mask = in1 ^ in2

    inner = fa[inner_mask]
    inner_cost = cost_smooth[inner_mask]
    band_idx = np.where(band)[0]
    nb = len(band_idx)
    idx_map = np.full(F, -1, dtype=np.int64)
    idx_map[band_idx] = np.arange(nb)

    if len(inner):
        g = sp.coo_matrix((np.ones(len(inner)),
                           (idx_map[inner[:, 0]], idx_map[inner[:, 1]])),
                          shape=(nb, nb))
        ncomp, comp = connected_components(g, directed=False)
    else:
        ncomp, comp = nb, np.arange(nb)


    order = np.argsort(comp, kind="stable")
    bounds = np.searchsorted(comp[order], np.arange(ncomp + 1))


    if len(inner):
        inner_comp = comp[idx_map[inner[:, 0]]]
        i_ord = np.argsort(inner_comp, kind="stable")
        i_bounds = np.searchsorted(inner_comp[i_ord], np.arange(ncomp + 1))
    else:
        i_ord = np.zeros(0, np.int64)
        i_bounds = np.zeros(ncomp + 1, np.int64)


    if cross_mask.any():
        ce = fa[cross_mask]
        cross_cost = cost_smooth[cross_mask]
        first_inside = band[ce[:, 0]]
        cross_inside = np.where(first_inside, ce[:, 0], ce[:, 1])
        cross_outside = np.where(first_inside, ce[:, 1], ce[:, 0])
        cross_comp = comp[idx_map[cross_inside]]
        c_ord = np.argsort(cross_comp, kind="stable")
        c_bounds = np.searchsorted(cross_comp[c_ord], np.arange(ncomp + 1))
    else:
        cross_inside = cross_outside = cross_cost = None
        c_ord = np.zeros(0, np.int64)
        c_bounds = np.zeros(ncomp + 1, np.int64)

    return {
        "band_idx": band_idx, "n_band": nb, "ncomp": ncomp,
        "order": order, "bounds": bounds,
        "inner": inner, "inner_cost": inner_cost,
        "i_ord": i_ord, "i_bounds": i_bounds,
        "cross_inside": cross_inside, "cross_outside": cross_outside,
        "cross_cost": cross_cost, "c_ord": c_ord, "c_bounds": c_bounds,
        "sub_pos": np.full(F, -1, dtype=np.int64),
    }


def _make_tasks(st, labels, unary_provider, label, F, min_component):
    """Build subproblem arguments for each component of a label."""
    tasks = []
    sub_pos = st["sub_pos"]
    for c in range(st["ncomp"]):
        sub = st["band_idx"][st["order"][st["bounds"][c]:st["bounds"][c + 1]]]
        if len(sub) < min_component:
            continue
        psub = labels[sub]
        if (psub == label).all():
            continue
        sub_pos[sub] = np.arange(len(sub))

        sel_i = st["i_ord"][st["i_bounds"][c]:st["i_bounds"][c + 1]]
        if len(sel_i):
            el = st["inner"][sel_i]
            el_local = np.stack([sub_pos[el[:, 0]], sub_pos[el[:, 1]]], axis=1)
            ec = st["inner_cost"][sel_i]
        else:
            el_local = np.zeros((0, 2), np.int64)
            ec = np.zeros(0)

        ba = np.zeros(len(sub)); bs = np.zeros(len(sub))
        if st["cross_inside"] is not None:
            sel_c = st["c_ord"][st["c_bounds"][c]:st["c_bounds"][c + 1]]
            if len(sel_c):
                pos = sub_pos[st["cross_inside"][sel_c]]
                olab = labels[st["cross_outside"][sel_c]]
                cc2 = st["cross_cost"][sel_c]
                is_lab = olab == label

                np.add.at(bs, pos[is_lab], cc2[is_lab])

                np.add.at(ba, pos[~is_lab], cc2[~is_lab])

        cost_alpha, cost_keep = unary_provider.pair_costs(sub, label, psub)
        cost_alpha = np.asarray(cost_alpha)
        cost_keep = np.asarray(cost_keep)
        if cost_alpha.shape != (len(sub),) or cost_keep.shape != (len(sub),):
            raise ValueError(
                "unary provider must return two [num_faces] arrays; "
                f"got {cost_alpha.shape} and {cost_keep.shape} for {len(sub)} faces"
            )
        if (np.isnan(cost_alpha).any() or np.isnan(cost_keep).any()
                or (cost_alpha < 0).any() or (cost_keep < 0).any()):
            raise ValueError("unary provider returned NaN or negative capacities")
        tasks.append((
            sub, label, psub, cost_alpha, cost_keep,
            el_local, ec, ba, bs,
        ))
        sub_pos[sub] = -1
    return tasks






def _return_with_optional_stats(labels, stats, return_stats):
    return (labels, stats) if return_stats else labels


def graph_cut_fast_from_unary(
        face_labels, unary_provider, mesh,
        _lambda=1.0, iterations=1, band_rings=2,
        smooth_mode="log", smooth_eps=1e-10,
        n_jobs=None, min_component=1, converge=False,
        max_sweeps=8, verbose=False,
        converge_min_change=0,
        max_band_label_product=0, max_band_faces=0,
        max_components=0, band_fallbacks=None,
        return_stats=False):
    """Run boundary-band alpha expansion with an on-demand unary provider.

    ``unary_provider.pair_costs(face_ids, alpha, keep_labels)`` returns two
    ``[len(face_ids)]`` cost arrays, allowing hard-label paths to use O(F) memory.

    band_rings: number of boundary rings; None or infinity uses the full graph.
    converge: repeat sweeps until labels stabilize; otherwise run ``iterations``.
    n_jobs: parallelism; defaults to ``min(32, cpu_count)``.
    max_band_label_product / max_band_faces / max_components:
                optional deterministic complexity budgets checked before graph build;
                zero disables the corresponding limit.
    band_fallbacks: narrower bands to try when a complexity budget is exceeded.
    """
    labels = np.asarray(face_labels).copy().astype(np.int64)
    labels_input = labels.copy()
    F = len(labels)
    active_labels = int(len(np.unique(labels[labels >= 0])))
    stats = {
        "applied": False,
        "reason": None,
        "budget_skipped": False,
        "requested_band_rings": (
            None if band_rings is None or band_rings == float("inf")
            else int(band_rings)
        ),
        "chosen_band_rings": (
            None if band_rings is None or band_rings == float("inf")
            else int(band_rings)
        ),
        "active_labels": active_labels,
        "subproblems": 0,
        "band_faces": 0,
        "components": 0,
        "sweeps": 0,
        "changed_faces": 0,
    }
    if _ig is None:
        print("[Warning] python-igraph is not installed; skipping graph cut. "
              "Install it with: pip install python-igraph")
        stats["reason"] = "igraph_missing"
        return _return_with_optional_stats(labels, stats, return_stats)
    fa = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(fa) == 0 or F == 0:
        stats["reason"] = "empty_mesh" if F == 0 else "no_face_adjacency"
        return _return_with_optional_stats(labels, stats, return_stats)
    if not hasattr(unary_provider, "pair_costs"):
        raise TypeError("unary_provider must implement pair_costs()")

    if n_jobs is None:
        n_jobs = min(32, os.cpu_count() or 8)

    full_graph = band_rings is None or band_rings == float("inf")


    nbr_index = None if full_graph else build_neighbor_index(mesh)




    has_budget = any((max_band_label_product, max_band_faces, max_components))
    if has_budget and not full_graph:


        candidates = [int(band_rings)]
        if band_fallbacks:
            for b in band_fallbacks:
                b = int(b)
                if 1 <= b < int(band_rings) and b not in candidates:
                    candidates.append(b)
            candidates = [candidates[0]] + sorted(candidates[1:], reverse=True)

        active_labels = int(len(np.unique(labels[labels >= 0])))
        chosen_band = None
        last_reasons = []
        for cand in candidates:
            preflight = _build_band_structure(
                labels, mesh, fa, np.zeros(len(fa), dtype=np.float32), F,
                cand, nbr_index=nbr_index)
            if preflight is None:
                if verbose:
                    print("  [GC-fast] preflight: no label boundary, skip",
                          flush=True)
                stats["reason"] = "no_label_boundary"
                return _return_with_optional_stats(labels, stats, return_stats)
            band_faces = int(preflight["n_band"])
            components = int(preflight["ncomp"])
            stats["band_faces"] = band_faces
            stats["components"] = components
            band_label_product = band_faces * active_labels
            reasons = []
            if (max_band_label_product
                    and band_label_product > max_band_label_product):
                reasons.append(
                    f"band*labels={band_label_product}>{max_band_label_product}"
                )
            if max_band_faces and band_faces > max_band_faces:
                reasons.append(f"band={band_faces}>{max_band_faces}")
            if max_components and components > max_components:
                reasons.append(f"components={components}>{max_components}")
            if verbose or reasons:
                print(
                    "  [GC-fast] preflight "
                    f"band={cand} band_faces={band_faces}/{F} "
                    f"labels={active_labels} components={components} "
                    f"band*labels={band_label_product}"
                    + ("" if not reasons else "  OVER: " + ", ".join(reasons)),
                    flush=True,
                )
            last_reasons = reasons
            if not reasons:
                chosen_band = cand
                break

        if chosen_band is None:
            print(
                "  [GC-fast] complexity budget exceeded at all bands "
                f"{candidates}; keep input labels ("
                + ", ".join(last_reasons) + ")",
                flush=True,
            )
            stats.update({
                "reason": "complexity_budget_exceeded",
                "budget_skipped": True,
                "budget_candidates": candidates,
                "budget_reasons": last_reasons,
            })
            return _return_with_optional_stats(labels, stats, return_stats)
        if chosen_band != int(band_rings):
            print(
                f"  [GC-fast] band downgraded {int(band_rings)} -> {chosen_band} "
                "to fit complexity budget",
                flush=True,
            )
            band_rings = chosen_band
        stats["chosen_band_rings"] = int(band_rings)

    cost_smooth = compute_smooth_costs(
        mesh, _lambda=_lambda, smooth_eps=smooth_eps,
        smooth_mode=smooth_mode,
    )


    n_sweeps = int(max_sweeps) if converge else int(iterations)

    for _it in range(n_sweeps):
        changed_total = 0
        uniq = np.unique(labels[labels >= 0])






        struct = None

        for label in uniq:
            if struct is None:
                struct = _build_band_structure(
                    labels, mesh, fa, cost_smooth, F,
                    None if full_graph else int(band_rings),
                    nbr_index=nbr_index)
                if struct is None:
                    break
                stats["band_faces"] = struct["n_band"]
                stats["components"] = struct["ncomp"]

            tasks = _make_tasks(struct, labels, unary_provider, int(label),
                                F, min_component)
            stats["subproblems"] += len(tasks)
            if not tasks:
                continue

            tasks.sort(key=lambda t: -len(t[0]))
            if n_jobs > 1 and len(tasks) > 1:
                with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                    results = list(ex.map(_solve_subproblem, tasks))
            else:
                results = [_solve_subproblem(t) for t in tasks]


            changed_here = 0
            for sub, assign in results:
                if assign.any():
                    sel = sub[assign]
                    changed_here += int((labels[sel] != label).sum())
                    labels[sel] = label
            changed_total += changed_here
            if changed_here:
                struct = None

        stats["sweeps"] = _it + 1
        if verbose:
            print("  [GC-fast] sweep=%d band=%d/%d (%.2f%%) subproblems=%d changed=%d" % (
                _it, stats["band_faces"], F, 100 * stats["band_faces"] / max(F, 1),
                stats["subproblems"], changed_total), flush=True)



        if changed_total <= converge_min_change:
            break

    stats["changed_faces"] = int(np.count_nonzero(labels != labels_input))
    stats["applied"] = bool(stats["subproblems"] > 0)
    if stats["changed_faces"]:
        stats["reason"] = "changed"
    elif stats["applied"]:
        stats["reason"] = "no_change"
    else:
        stats["reason"] = "no_subproblems"
    return _return_with_optional_stats(labels, stats, return_stats)


def graph_cut_fast(face_labels, face_logits_sum, face_point_count, mesh,
                   _lambda=1.0, iterations=1, band_rings=2,
                   smooth_mode="log", smooth_eps=1e-10,
                   n_jobs=None, min_component=1, converge=False,
                   max_sweeps=8, verbose=False,
                   converge_min_change=0,
                   max_band_label_product=0, max_band_faces=0,
                   max_components=0, band_fallbacks=None,
                   return_stats=False):
    """Compatibility layer for the legacy logits API.

    Returns labels when ``return_stats=False`` and ``(labels, stats)`` otherwise.
    """
    provider = LogitsUnaryCostProvider(face_logits_sum, face_point_count)
    return graph_cut_fast_from_unary(
        face_labels,
        provider,
        mesh,
        _lambda=_lambda,
        iterations=iterations,
        band_rings=band_rings,
        smooth_mode=smooth_mode,
        smooth_eps=smooth_eps,
        n_jobs=n_jobs,
        min_component=min_component,
        converge=converge,
        max_sweeps=max_sweeps,
        verbose=verbose,
        converge_min_change=converge_min_change,
        max_band_label_product=max_band_label_product,
        max_band_faces=max_band_faces,
        max_components=max_components,
        band_fallbacks=band_fallbacks,
        return_stats=return_stats,
    )
