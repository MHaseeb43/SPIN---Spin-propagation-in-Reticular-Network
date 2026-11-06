"""
Small helpers shared across the pipeline.
"""

import os
import numpy as np
import networkx as nx
from skimage.draw import line as draw_line


# sknw occasionally produces "phantom" edges when stitching tiles: the
# straight-line distance between the two endpoints is much longer than the
# skeleton path the edge supposedly follows. We treat an edge as phantom when
#     endpoint distance > PHANTOM_GUARD_MULTIPLIER × path length + padding.
PHANTOM_GUARD_MULTIPLIER = 3.0
PHANTOM_GUARD_PADDING_PX = 5.0


def is_phantom_edge(endpoint_dist_px: float, length_um: float, pixel_size_um: float) -> bool:
    path_px = length_um / pixel_size_um
    return endpoint_dist_px > path_px * PHANTOM_GUARD_MULTIPLIER + PHANTOM_GUARD_PADDING_PX


def save_graphml(G: nx.Graph, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nx.write_graphml(G, path)


def render_graph_as_skeleton(G: nx.Graph, shape: tuple,
                             pixel_size_um: float | None = None) -> np.ndarray:
    """
    Draw every edge of G as a straight 1-px line. Returns a uint8 0/1 image.

    If `pixel_size_um` is given, phantom edges are skipped so they don't show
    up as long lines across the image.
    """
    skel = np.zeros(shape, dtype=np.uint8)
    H, W = shape
    for u, v, edata in G.edges(data=True):
        nd_u, nd_v = G.nodes[u], G.nodes[v]
        if "o_row" not in nd_u or "o_row" not in nd_v:
            continue
        r0 = max(0, min(int(round(nd_u["o_row"])), H - 1))
        c0 = max(0, min(int(round(nd_u["o_col"])), W - 1))
        r1 = max(0, min(int(round(nd_v["o_row"])), H - 1))
        c1 = max(0, min(int(round(nd_v["o_col"])), W - 1))
        if pixel_size_um and "length_um" in edata:
            if is_phantom_edge(np.hypot(r1 - r0, c1 - c0), edata["length_um"], pixel_size_um):
                continue
        rr, cc = draw_line(r0, c0, r1, c1)
        skel[rr, cc] = 1
    return skel
