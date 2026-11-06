"""
Graph refinement: turn the raw sknw graph into a clean conduit network.

The raw graph straight out of sknw is full of skeletonisation artefacts:
tiny stubs, duplicated junctions, chains of pass-through nodes, small
triangles at thick branch points and lots of disconnected debris. The
refinement runs ten deterministic passes, in this order:

   1. Remove phantom edges      (sknw tile-stitching artefacts)
   2. Merge close nodes         (duplicate junctions within a few µm)
   3. Tip pruning               (very short dead-end stubs)
   4. Degree-2 dissolve         (pass-through nodes on near-straight paths)
   5. Junction contraction      (tiny loops/triangles at thick branch points)
   6. Dead-end pruning          (longer stubs, now that edge lengths are merged)
   7. Component bridging        (optional, joins aligned nearby fragments)
   8. Node-count filter         (small fragments, relative to the largest)
   9. Length filter             (fragments too short to carry a signal)
  10. Keep ranked component(s)  (keep only the component(s) we analyse)

We use plain graph operations instead of a learned approach (e.g. a GNN
edge classifier) because the input is already binarised and skeletonised.
The remaining noise is structural, which simple rules handle reliably and
transparently, without any labelled training data.

All iterative passes are queue based, so each node is revisited only when
one of its neighbours changes. That keeps them roughly O(V + E) instead of
rescanning the whole graph on every iteration.
"""

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

from spin.analyser import analyse_component
from spin.utils import PHANTOM_GUARD_MULTIPLIER, is_phantom_edge


# ── Small geometry helpers ───────────────────────────────────────────────────

def _position(G: nx.Graph, node) -> np.ndarray:
    nd = G.nodes[node]
    return np.array([nd.get("o_row", 0.0), nd.get("o_col", 0.0)], dtype=float)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two 2-D vectors."""
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def _local_direction(G: nx.Graph, node) -> np.ndarray | None:
    """
    Unit vector pointing away from the mean position of the node's neighbours.

    For a dead-end (degree 1) this is simply the direction the stub points in.
    Returns None for isolated nodes or when the direction is undefined.
    """
    nbrs = list(G.neighbors(node))
    if not nbrs:
        return None
    inward = np.mean([_position(G, nb) for nb in nbrs], axis=0) - _position(G, node)
    norm = np.linalg.norm(inward)
    if norm < 1e-9:
        return None
    return -inward / norm


# ── Node merging (shared by passes 2 and 5) ──────────────────────────────────

def _group_nodes(G: nx.Graph, pairs) -> dict:
    """
    Union-find over `pairs`; returns {representative: [member nodes]}.

    When two groups are joined, the representative is picked by node degree
    (ties broken by node ID), so the result does not depend on pair order.
    """
    parent = {n: n for n in G.nodes()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        da, db = G.degree(ra), G.degree(rb)
        if da < db or (da == db and ra < rb):
            parent[rb] = ra
        else:
            parent[ra] = rb

    groups = {}
    for n in G.nodes():
        groups.setdefault(find(n), []).append(n)
    return {rep: members for rep, members in groups.items() if len(members) > 1}


def _collapse_groups(G: nx.Graph, groups: dict) -> nx.Graph:
    """
    Replace every group of nodes by a single node at the group's centroid.

    Edges leaving the group are re-attached to the surviving node. If that
    creates a duplicate edge, the shorter of the two is kept. Edges inside
    the group disappear with the merged nodes.
    """
    pos = {n: _position(G, n) for members in groups.values() for n in members}

    for rep, members in groups.items():
        keep = rep if rep in G else members[0]
        centroid = np.mean([pos[m] for m in members], axis=0)
        G.nodes[keep]["o_row"] = float(centroid[0])
        G.nodes[keep]["o_col"] = float(centroid[1])

        for m in members:
            if m == keep or m not in G:
                continue
            for nbr, edata in list(G[m].items()):
                if nbr in members:
                    continue
                if G.has_edge(keep, nbr):
                    if edata.get("length_um", 0) < G[keep][nbr].get("length_um", 0):
                        G[keep][nbr].update(edata)
                else:
                    G.add_edge(keep, nbr, **edata)
            G.remove_node(m)
    return G


# ── Preparation ──────────────────────────────────────────────────────────────

def _to_simple_graph(G: nx.Graph) -> nx.Graph:
    """
    Convert the sknw MultiGraph into a plain Graph (always returns a copy).
    Where two nodes are joined by parallel edges, the shorter one is kept.
    """
    if not isinstance(G, nx.MultiGraph):
        return G.copy()

    simple = nx.Graph()
    simple.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if simple.has_edge(u, v):
            if data.get("length_um", 0) < simple[u][v].get("length_um", 0):
                simple[u][v].update(data)
        else:
            simple.add_edge(u, v, **data)
    return simple


def _remove_zero_length_edges(G: nx.Graph) -> nx.Graph:
    """sknw sometimes links two junction pixels at the same coordinate."""
    zero_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("length_um", 1) == 0]
    G.remove_edges_from(zero_edges)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


# ── Pass 1: phantom edges ────────────────────────────────────────────────────

def _remove_phantom_edges(G: nx.Graph, pixel_size_um: float) -> tuple[nx.Graph, int]:
    """
    Drop edges whose straight-line endpoint distance is far longer than the
    traced skeleton path. Such an edge cannot be a real conduit. It comes
    from tile stitching in sknw, and left in place it would make separate
    fragments look like one connected network.
    """
    if not pixel_size_um:
        return G, 0

    phantom_edges = []
    for u, v, d in G.edges(data=True):
        nu, nv = G.nodes[u], G.nodes[v]
        if "o_row" not in nu or "o_row" not in nv:
            continue
        dist_px = np.hypot(nv["o_row"] - nu["o_row"], nv["o_col"] - nu["o_col"])
        if is_phantom_edge(dist_px, d.get("length_um", 0), pixel_size_um):
            phantom_edges.append((u, v))

    G.remove_edges_from(phantom_edges)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G, len(phantom_edges)


# ── Pass 2: merge close nodes ────────────────────────────────────────────────

def _merge_close_nodes(G: nx.Graph, max_distance_um: float, pixel_size_um: float) -> nx.Graph:
    """Merge nodes that lie within `max_distance_um` of each other."""
    if G.number_of_nodes() == 0:
        return G

    # Node positions are stored in pixels, so convert the threshold first.
    max_distance_px = max_distance_um / pixel_size_um

    nodes = list(G.nodes())
    positions = np.array([_position(G, n) for n in nodes])
    pairs = [(nodes[i], nodes[j])
             for i, j in cKDTree(positions).query_pairs(max_distance_px)]
    if not pairs:
        return G

    return _collapse_groups(G, _group_nodes(G, pairs))


# ── Passes 3 and 6: prune short dead-end stubs ───────────────────────────────

def _prune_short_tips(G: nx.Graph, max_length_um: float) -> nx.Graph:
    """
    Repeatedly remove dead-end nodes (degree 1) whose edge is shorter than
    `max_length_um`. When a stub is removed, its neighbour may become a new
    dead end, so it goes back on the queue.

    Used twice: once early with a small threshold to clear sub-pixel stubs
    (pass 3), and again after degree-2 dissolve, when edge lengths reflect
    whole branches, with a larger threshold (pass 6).
    """
    tips = {n for n in G.nodes() if G.degree(n) == 1}

    while tips:
        node = tips.pop()
        if node not in G or G.degree(node) != 1:
            continue
        nbr = next(iter(G.neighbors(node)))
        if G[node][nbr].get("length_um", 0) >= max_length_um:
            continue
        G.remove_node(node)
        if nbr in G:
            if G.degree(nbr) == 1:
                tips.add(nbr)
            elif G.degree(nbr) == 0:
                G.remove_node(nbr)

    G.remove_nodes_from(list(nx.isolates(G)))
    return G


# ── Pass 4: degree-2 dissolve ────────────────────────────────────────────────

def _dissolve_degree2_nodes(G: nx.Graph, angle_threshold_deg: float) -> nx.Graph:
    """
    Remove pass-through nodes (degree 2) that sit on a nearly straight path,
    i.e. where the angle at the node is within `angle_threshold_deg` of 180°.
    The two edges are replaced by one edge with the combined length. Nodes
    at real bends are kept so the geometry of the network is preserved.
    """
    candidates = {n for n in G.nodes() if G.degree(n) == 2}

    while candidates:
        node = candidates.pop()
        if node not in G or G.degree(node) != 2:
            continue
        nbrs = list(G.neighbors(node))
        if len(nbrs) != 2:      # degree 2 can also be a single self-loop
            continue
        a, b = nbrs
        v1 = _position(G, a) - _position(G, node)
        v2 = _position(G, b) - _position(G, node)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            continue
        if abs(_angle_between(v1, v2) - 180) >= angle_threshold_deg:
            continue
        # Don't create a parallel edge (that would close a 3-node loop).
        if G.has_edge(a, b):
            continue

        combined = G[node][a].get("length_um", 0) + G[node][b].get("length_um", 0)
        G.add_edge(a, b, length_um=combined)
        G.remove_node(node)
        for nbr in (a, b):
            if G.degree(nbr) == 2:
                candidates.add(nbr)

    return G


# ── Pass 5: contract short junction-to-junction edges ────────────────────────

def _contract_short_junction_edges(G: nx.Graph, max_length_um: float) -> tuple[nx.Graph, int]:
    """
    Collapse very short edges between two non-terminal nodes (degree >= 2).

    A single thick branch point often ends up as several neighbouring nodes
    joined by tiny edges, frequently forming a small triangle. Tip pruning
    can't touch these because none of the nodes is a dead end. Grouping the
    short edges with union-find lets a whole cluster collapse into one node.

    Short self-loops are a related sknw artefact. They have no second node
    to merge into, so they are simply removed.

    Returns the graph and the number of clusters/self-loops removed.
    """
    if G.number_of_nodes() == 0:
        return G, 0

    short_self_loops = [(u, v) for u, v, d in G.edges(data=True)
                        if u == v and d.get("length_um", 0) < max_length_um]
    G.remove_edges_from(short_self_loops)

    short_edges = [(u, v) for u, v, d in G.edges(data=True)
                   if u != v and d.get("length_um", 0) < max_length_um
                   and G.degree(u) >= 2 and G.degree(v) >= 2]
    groups = _group_nodes(G, short_edges) if short_edges else {}
    if not groups:
        return G, len(short_self_loops)

    G = _collapse_groups(G, groups)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G, len(groups) + len(short_self_loops)


# ── Pass 7: bridge nearby components ─────────────────────────────────────────

def _bridge_close_components(
    G: nx.Graph,
    top_n: int,
    max_distance_um: float,
    pixel_size_um: float,
    max_angle_deg: float = 45.0,
) -> tuple[nx.Graph, int]:
    """
    Join the `top_n` largest components with synthetic edges when they are
    close and look like continuations of each other.

    For each pair of components we look at node pairs (one from each) within
    `max_distance_um`, closest first, and accept the first pair where both
    nodes' local directions point at each other within `max_angle_deg`. This
    keeps us from connecting fragments just because they happen to be close.
    After a merge the search restarts, so a chain of fragments can end up as
    one component.

    Bridge edges get `bridge=True` and their real length, so they can always
    be told apart from skeleton-derived edges later on.
    """
    if max_distance_um <= 0 or top_n < 2:
        return G, 0

    max_distance_px = max_distance_um / pixel_size_um
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    groups = [list(c) for c in components[:top_n]]
    n_bridges = 0

    def find_bridge(group_a, group_b):
        pos_a = np.array([_position(G, n) for n in group_a])
        pos_b = np.array([_position(G, n) for n in group_b])
        dir_a = [_local_direction(G, n) for n in group_a]
        dir_b = [_local_direction(G, n) for n in group_b]
        tree_b = cKDTree(pos_b)

        candidates = sorted(
            ((float(np.linalg.norm(p - pos_b[j])), i, j)
             for i, p in enumerate(pos_a)
             for j in tree_b.query_ball_point(p, r=max_distance_px)),
            key=lambda c: c[0],
        )
        for dist_px, i, j in candidates:
            if dir_a[i] is None or dir_b[j] is None or dist_px < 1e-9:
                continue
            bridge_unit = (pos_b[j] - pos_a[i]) / dist_px
            if (_angle_between(dir_a[i], bridge_unit) <= max_angle_deg
                    and _angle_between(dir_b[j], -bridge_unit) <= max_angle_deg):
                return group_a[i], group_b[j], dist_px
        return None

    merged = True
    while merged and len(groups) > 1:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                bridge = find_bridge(groups[i], groups[j])
                if bridge is None:
                    continue
                u, v, dist_px = bridge
                G.add_edge(u, v, length_um=dist_px * pixel_size_um, weight=dist_px, bridge=True)
                n_bridges += 1
                groups[i] = groups[i] + groups[j]
                del groups[j]
                merged = True
                break
            if merged:
                break

    return G, n_bridges


# ── Pass 8: node-count filter ────────────────────────────────────────────────

def _remove_small_components(
    G: nx.Graph,
    min_component_nodes: int,
    min_component_fraction: float,
) -> tuple[nx.Graph, int]:
    """
    Remove components smaller than
        max(min_component_nodes, min_component_fraction × size of largest).
    Returns the graph and the threshold actually used.
    """
    components = list(nx.connected_components(G))
    if not components:
        return G, min_component_nodes

    largest_size = max(len(c) for c in components)
    threshold = max(min_component_nodes, int(min_component_fraction * largest_size))

    G.remove_nodes_from([n for c in components if len(c) < threshold for n in c])
    return G, threshold


# ── Pass 9: total-length filter ──────────────────────────────────────────────

def _remove_short_components(G: nx.Graph, min_total_length_um: float) -> tuple[nx.Graph, int]:
    """
    Remove components whose summed edge length is below `min_total_length_um`
    (rationale: Kelch et al. 2015, eLife 4:e06493).
    Returns the graph and the number of components removed.
    """
    short = []
    for component in nx.connected_components(G):
        total_length = sum(d.get("length_um", 0)
                           for _, _, d in G.subgraph(component).edges(data=True))
        if total_length < min_total_length_um:
            short.append(component)

    G.remove_nodes_from([n for c in short for n in c])
    return G, len(short)


# ── Pass 10: keep component(s) by size rank ──────────────────────────────────

def _keep_ranked_components(G: nx.Graph, ranks) -> tuple[nx.Graph, int]:
    """
    Keep only the components at the given size ranks (1 = largest).

    `ranks` can be a single int or a list such as [1, 2, 3]. Ranks that don't
    exist are ignored; if none of them exist, the largest component is kept.
    Returns the graph and the number of components removed.
    """
    if isinstance(ranks, int):
        ranks = (ranks,)

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(components) <= 1:
        return G, 0

    keep_idx = {r - 1 for r in ranks if 1 <= r <= len(components)} or {0}
    G.remove_nodes_from([n for i, c in enumerate(components) if i not in keep_idx for n in c])
    return G, len(components) - len(keep_idx)


# ── Public API ───────────────────────────────────────────────────────────────

def refine_graph(
    G: nx.Graph,
    min_length_um: float,
    angle_threshold_deg: float,
    min_component_nodes: int = 20,
    min_component_fraction: float = 0.005,
    dead_end_length_um: float = 15.0,
    min_component_length_um: float = 50.0,
    max_merge_distance_um: float = 2.0,
    junction_contraction_length_um: float = 4.0,
    keep_largest_component_only: bool = True,
    component_rank_to_keep: int | list[int] = 1,
    component_bridge_top_n: int = 3,
    component_bridge_max_distance_um: float = 0.0,
    component_bridge_max_angle_deg: float = 45.0,
    pixel_size_um: float | None = None,
    component_analysis_k_sample: int = 200,
    on_pass=None,
) -> tuple[nx.Graph, list[dict], nx.Graph | None]:
    """
    Run all ten refinement passes on a raw sknw graph.

    The input graph is not modified. All thresholds are documented in
    config.py, which is where the pipeline takes them from.

    Parameters (beyond the thresholds)
    ----------------------------------
    keep_largest_component_only : if True, pass 10 keeps only the
        component(s) listed in `component_rank_to_keep`. The top-3
        components are analysed just before that, so their statistics end
        up in the metrics even though they are removed from the graph.
    component_bridge_max_distance_um : 0 disables bridging (pass 7).
    pixel_size_um : µm per pixel. Needed for the phantom-edge check and to
        convert µm thresholds to pixel distances.
    component_analysis_k_sample : number of sampled source nodes for the
        shortest-path / betweenness estimates of the top-3 components.
    on_pass : optional callback `on_pass(number, name, G_before, G_after)`
        called after every pass. Used by `pipeline.py --debug-passes` to
        draw before/after images. When None, no graph copies are made.

    Returns
    -------
    G : the refined graph.
    steps : one dict per pass with node/edge counts and the thresholds used.
        The last entry also holds "pre_prune_top_components".
    G_pre_prune : the graph just before pass 10 (all components still
        present), or None when keep_largest_component_only is False.
    """
    G = _remove_zero_length_edges(_to_simple_graph(G))
    pixel_size = pixel_size_um or 1.0

    steps = [{"pass": "initial",
              "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
              "removed_nodes": 0, "removed_edges": 0}]
    print(f"  {'Raw graph':<22}: {G.number_of_nodes():>7,} nodes  {G.number_of_edges():>7,} edges")

    def record(number, name, label, before, note="", **details):
        """Store counts for a finished pass, print a log line and call on_pass."""
        n, e = G.number_of_nodes(), G.number_of_edges()
        prev = steps[-1]
        step = {"pass": name, "n_nodes": n, "n_edges": e,
                "removed_nodes": prev["n_nodes"] - n,
                "removed_edges": prev["n_edges"] - e}
        step.update(details)
        steps.append(step)
        print(f"  {label:<22}: {n:>7,} nodes  {e:>7,} edges"
              f"  ({n - prev['n_nodes']:+,} N, {e - prev['n_edges']:+,} E){note}")
        if on_pass:
            on_pass(number, name, before, G)

    def snapshot():
        return G.copy() if on_pass else None

    # 1. Phantom edges
    before = snapshot()
    G, n_phantom = _remove_phantom_edges(G, pixel_size_um)
    record(1, "remove_phantom_edges", "Phantom edges removed", before,
           guard_multiplier=PHANTOM_GUARD_MULTIPLIER)

    # 2. Merge duplicate junctions
    before = snapshot()
    G = _merge_close_nodes(G, max_merge_distance_um, pixel_size)
    record(2, "merge_close_nodes", "Close nodes merged", before,
           f"  within {max_merge_distance_um} um", threshold_um=max_merge_distance_um)

    # 3. Remove tiny stubs
    before = snapshot()
    G = _prune_short_tips(G, min_length_um)
    record(3, "tip_prune", "Tips pruned", before,
           f"  stubs < {min_length_um} um", threshold_um=min_length_um)

    # 4. Dissolve straight pass-through nodes
    before = snapshot()
    G = _dissolve_degree2_nodes(G, angle_threshold_deg)
    record(4, "degree2_dissolve", "Degree-2 dissolved", before,
           f"  within {angle_threshold_deg} deg of straight",
           angle_threshold_deg=angle_threshold_deg)

    # 5. Collapse loop/triangle clusters. After a contraction, re-dissolving
    #    can create a new short edge, so repeat a few times until stable.
    before = snapshot()
    n_contracted = 0
    for _ in range(5):
        G, n_round = _contract_short_junction_edges(G, junction_contraction_length_um)
        n_contracted += n_round
        if n_round == 0:
            break
        G = _dissolve_degree2_nodes(G, angle_threshold_deg)
    record(5, "junction_edge_contraction", "Junctions contracted", before,
           f"  {n_contracted:,} clusters < {junction_contraction_length_um} um",
           merged_clusters=n_contracted, threshold_um=junction_contraction_length_um)

    # 6. Remove longer dead-end stubs (lengths are now whole branches)
    before = snapshot()
    G = _prune_short_tips(G, dead_end_length_um)
    record(6, "dead_end_prune", "Dead ends pruned", before,
           f"  stubs < {dead_end_length_um} um", threshold_um=dead_end_length_um)

    # 7. Bridge nearby components. This runs before the size filters so that
    #    a small piece cut off by a skeletonisation gap can rejoin the main
    #    network instead of being thrown away for being too small on its own.
    before = snapshot()
    G, n_bridges = _bridge_close_components(
        G, component_bridge_top_n, component_bridge_max_distance_um, pixel_size,
        max_angle_deg=component_bridge_max_angle_deg,
    )
    record(7, "bridge_close_components", "Components bridged", before,
           f"  {n_bridges} bridge(s) <= {component_bridge_max_distance_um} um, "
           f"aligned within {component_bridge_max_angle_deg} deg",
           removed_nodes=0, removed_edges=0,
           bridges_added=n_bridges,
           top_n_considered=component_bridge_top_n,
           max_distance_um=component_bridge_max_distance_um,
           max_angle_deg=component_bridge_max_angle_deg)

    # 8. Remove small fragments
    before = snapshot()
    G, node_threshold = _remove_small_components(G, min_component_nodes, min_component_fraction)
    record(8, "node_count_filter", "Small comps removed", before,
           f"  < {node_threshold} nodes", adaptive_threshold_nodes=int(node_threshold))

    # 9. Remove fragments with too little total length
    before = snapshot()
    G, n_short = _remove_short_components(G, min_component_length_um)
    record(9, "component_length_filter", "Short comps removed", before,
           f"  {n_short} comps < {min_component_length_um} um total",
           removed_components=n_short, threshold_um=min_component_length_um)

    # Last chance to look at the other components before pass 10 drops them.
    pre_prune_top_components = None
    G_pre_prune = None
    if keep_largest_component_only:
        G_pre_prune = G.copy()
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        n_total = G.number_of_nodes()
        pre_prune_top_components = [
            analyse_component(G.subgraph(components[i]).copy(), n_total,
                              component_analysis_k_sample, rank=i + 1)
            for i in range(min(3, len(components)))
        ]

    # 10. Keep only the selected component(s)
    before = snapshot()
    n_dropped = 0
    if keep_largest_component_only:
        G, n_dropped = _keep_ranked_components(G, component_rank_to_keep)
    record(10, "keep_component_by_rank", "Ranked comps kept", before,
           f"  rank {component_rank_to_keep}" if keep_largest_component_only else "  (disabled)",
           removed_components=n_dropped,
           enabled=keep_largest_component_only,
           rank_kept=component_rank_to_keep,
           pre_prune_top_components=pre_prune_top_components)

    n0, e0 = steps[0]["n_nodes"], steps[0]["n_edges"]
    print(f"  {'Total reduction':<22}: {100 * (1 - G.number_of_nodes() / max(n0, 1)):.1f}% nodes, "
          f"{100 * (1 - G.number_of_edges() / max(e0, 1)):.1f}% edges")

    return G, steps, G_pre_prune
