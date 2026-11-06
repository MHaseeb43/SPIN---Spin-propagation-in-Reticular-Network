"""
Graph construction: convert a 1-px skeleton into a NetworkX graph with sknw.

Nodes are junctions and end points of the skeleton; edges are the skeleton
paths between them, with their physical length in µm.

Two entry points:
  build_graph()        runs sknw directly; fine for small skeleton pieces.
  build_tiled_graph()  splits a large piece into overlapping tiles, runs
                       sknw on each tile and stitches the results together.

We tile instead of downsampling because the skeleton is only one pixel
wide. Decimating it (e.g. taking every 4th pixel) breaks the lines apart
and erases most branch points before sknw ever sees them.
"""

import itertools
from collections import defaultdict

import numpy as np
import networkx as nx
import sknw
from scipy.spatial import cKDTree


def build_graph(skeleton: np.ndarray, pixel_size_um: float) -> nx.Graph:
    """
    Run sknw on `skeleton` and clean up the attributes so the graph can be
    written to GraphML.

    Node attributes:  o_row, o_col  (position in array coordinates, float)
    Edge attributes:  length_um     (path length in µm)
                      weight        (path length in pixels, from sknw)
                      pts           (pixel path, stored as a string)
    """
    graph = sknw.build_sknw(skeleton)

    for _, _, data in graph.edges(data=True):
        data["length_um"] = float(data.get("weight", 0)) * pixel_size_um
        if "pts" in data:
            pts = data["pts"]
            data["pts"] = str(pts.tolist() if isinstance(pts, np.ndarray) else pts)
        if "weight" in data:
            data["weight"] = float(data["weight"])

    for _, data in graph.nodes(data=True):
        if "o" in data:
            o = np.asarray(data.pop("o"))
            data["o_row"] = float(o[0])
            data["o_col"] = float(o[1])
        if "pts" in data:
            pts = data["pts"]
            data["pts"] = str(pts.tolist() if isinstance(pts, np.ndarray) else pts)

    return graph


def _union_find(node_list, pairs) -> dict:
    """Group nodes connected by `pairs`. Returns {node: representative}."""
    parent = {n: n for n in node_list}
    rank = {n: 0 for n in node_list}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:          # path compression
            parent[x], x = root, parent[x]
        return root

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    return {n: find(n) for n in node_list}


def _merge_nodes_kdtree(G: nx.Graph, radius: float) -> nx.Graph:
    """
    Merge all nodes within `radius` pixels of each other into one node at
    their centroid. Used to remove the duplicate junctions that appear when
    the same junction lies in the overlap of two tiles.

    Self-loops created by the merge are dropped. When two edges end up
    between the same pair of nodes, the longer one is kept.
    Returns a new graph; G itself is not modified.
    """
    if G.number_of_nodes() < 2:
        return G.copy()

    nodes = list(G.nodes())
    positions = np.array([[float(G.nodes[n].get("o_row", 0.0)),
                           float(G.nodes[n].get("o_col", 0.0))] for n in nodes])

    pairs = [(nodes[i], nodes[j]) for i, j in cKDTree(positions).query_pairs(r=radius)]
    canon = _union_find(nodes, pairs)

    members = defaultdict(list)
    for i, n in enumerate(nodes):
        members[canon[n]].append(i)

    merged = nx.Graph()
    for rep, idx in members.items():
        row, col = positions[idx].mean(axis=0)
        merged.add_node(rep, o_row=round(float(row), 3), o_col=round(float(col), 3))

    for u, v, edata in G.edges(data=True):
        cu, cv = canon[u], canon[v]
        if cu == cv:
            continue
        if merged.has_edge(cu, cv):
            if float(edata.get("length_um", 0.0)) > float(merged[cu][cv].get("length_um", 0.0)):
                merged[cu][cv].update(edata)
        else:
            merged.add_edge(cu, cv, **edata)

    return merged


def build_tiled_graph(
    skeleton_crop: np.ndarray,
    r_origin: int,
    c_origin: int,
    pixel_size_um: float,
    tile_size: int = 2048,
    overlap: int = 256,
    merge_radius: float = 3.0,
) -> nx.Graph:
    """
    Build the graph of a large skeleton piece tile by tile, at full resolution.

    Tiles of `tile_size` px overlap their neighbours by `overlap` px, so a
    junction near a tile border is seen completely in at least one tile.
    Duplicates from the overlap zones are merged afterwards
    (`_merge_nodes_kdtree`). Segments between junctions are typically
    ~15 px long, so a 256-px overlap is plenty; any leftover stubs at tile
    borders are removed later by the refinement.

    Parameters
    ----------
    skeleton_crop      : 2-D skeleton of the piece (bool or uint8)
    r_origin, c_origin : top-left corner of the crop in the full image.
                         Returned node positions are in full-image coordinates.
    pixel_size_um      : µm per pixel
    tile_size, overlap : tile geometry in pixels
    merge_radius       : max distance in px for merging duplicate nodes
    """
    H, W = skeleton_crop.shape
    stride = max(1, tile_size - overlap)

    def tile_starts(size):
        # Regular grid, plus one last tile aligned to the far edge if needed.
        starts = list(range(0, size, stride))
        if not starts or starts[-1] + tile_size < size:
            starts.append(max(0, size - tile_size))
        return sorted(set(starts))

    row_starts, col_starts = tile_starts(H), tile_starts(W)
    n_tiles = len(row_starts) * len(col_starts)
    print(f"  Tiled graph: {H}x{W} crop  |  {len(row_starts)}x{len(col_starts)}={n_tiles} tiles  "
          f"(size={tile_size}, overlap={overlap})")

    combined = nx.Graph()
    next_id = itertools.count()
    n_empty = 0

    for tr0 in row_starts:
        for tc0 in col_starts:
            tile = skeleton_crop[tr0:tr0 + tile_size, tc0:tc0 + tile_size]
            if not tile.any():
                n_empty += 1
                continue

            try:
                tile_g = build_graph(tile.astype(np.uint8), pixel_size_um)
            except Exception:
                continue
            if tile_g.number_of_nodes() == 0:
                continue

            # Tile coordinates -> full-image coordinates
            for _, data in tile_g.nodes(data=True):
                if "o_row" in data:
                    data["o_row"] = float(data["o_row"]) + tr0 + r_origin
                    data["o_col"] = float(data["o_col"]) + tc0 + c_origin

            # Node IDs must be unique across tiles before we combine them
            combined.update(nx.relabel_nodes(tile_g, {n: next(next_id) for n in tile_g.nodes()}))

    if combined.number_of_nodes() == 0:
        return combined

    result = _merge_nodes_kdtree(combined, merge_radius)
    print(f"  After tile merge: {result.number_of_nodes():,}N  {result.number_of_edges():,}E  "
          f"({n_tiles - n_empty}/{n_tiles} non-empty tiles)")
    return result
