"""
SPIN: Signal Propagation in the Reticular Network of Human Lymph Nodes.

Turns a binarised confocal Z-stack of lymph node tissue into a graph of the
reticular (conduit) network and measures it.

    1. Max projection   collapse all Z-layers into one 2-D binary image
    2. Skeletonisation  thin the network down to 1-px-wide centre lines
    3. Graph building   junctions become nodes, conduit segments edges (sknw)
    4. Refinement       remove skeletonisation artefacts (10 passes)
    5. Analysis         network metrics, figures and a metrics JSON

Usage
-----
    python pipeline.py --all                          # every image in data/Input/
    python pipeline.py --image <path>                 # a single image
    python pipeline.py --image <path> --debug-passes  # + before/after image per refinement pass

Output (per image, in data/output/<image name>/)
------------------------------------------------
    Refined Graph/  skeleton, raw graph and refined graph (GraphML + images)
    Metrics/        metrics_maxproj.json and the full console log
    Comparison/     4-panel overview of the pipeline stages, timing chart
After --all, data/output/spin_summary.csv holds one row per image.
"""

import argparse
import contextlib
import csv
import gc
import json
import os
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tifffile
from skimage.draw import disk as draw_disk, line as draw_line
from skimage.measure import label as sk_label, regionprops
from skimage.morphology import dilation, disk

import config
from spin.analyser import analyse
from spin.gpu_utils import cucim_available, gpu_available
from spin.graph_builder import build_graph, build_tiled_graph
from spin.graph_refiner import refine_graph
from spin.loader import load_layer, open_stack
from spin.skeletoniser import compute_skeleton
from spin.utils import is_phantom_edge, render_graph_as_skeleton, save_graphml

# Windows consoles default to a legacy code page; make sure µm etc. print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── Console helpers ──────────────────────────────────────────────────────────

class StepTimer:
    """Measures wall-clock time per pipeline step and prints a summary table."""

    def __init__(self):
        self.times = {}                 # step label -> seconds, in run order
        self._start = None

    def start(self, label: str):
        self._label, self._start = label, time.perf_counter()

    def stop(self) -> float:
        elapsed = time.perf_counter() - self._start
        self.times[self._label] = elapsed
        return elapsed

    def report(self):
        if not self.times:
            return
        total = sum(self.times.values())
        width = max(len(k) for k in self.times) + 2
        print(f"\n{'─' * 55}\n  {'STEP':<{width}}  {'TIME':>9}   {'%':>5}\n{'─' * 55}")
        for label, secs in self.times.items():
            print(f"  {label:<{width}}  {_format_duration(secs):>9}   {100 * secs / total:>4.1f}%")
        print(f"{'─' * 55}\n  {'TOTAL':<{width}}  {_format_duration(total):>9}   100.0%\n{'─' * 55}\n")


def _format_duration(secs: float) -> str:
    mins = int(secs // 60)
    return f"{mins}m {secs % 60:04.1f}s" if mins else f"{secs:.1f}s"


class _Tee:
    """File-like object that writes to several streams (console + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


# ── Paths ────────────────────────────────────────────────────────────────────

def _list_input_images() -> list[str]:
    if not os.path.isdir(config.INPUT_DIR):
        raise FileNotFoundError(f"Input folder not found: {config.INPUT_DIR}")
    return sorted(os.path.join(config.INPUT_DIR, f) for f in os.listdir(config.INPUT_DIR)
                  if f.lower().endswith((".tif", ".tiff")))


def _image_output_dirs(image_path: str) -> tuple[dict, str]:
    """Create (if needed) and return the output folders for one image."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    if stem.endswith(".ome"):
        stem = stem[:-4]

    base = os.path.join(config.OUTPUT_DIR, stem)
    dirs = {
        "base":       base,
        "refined":    os.path.join(base, "Refined Graph"),
        "metrics":    os.path.join(base, "Metrics"),
        "comparison": os.path.join(base, "Comparison"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs, stem


# ── Steps 1 and 3: max projection and raw graph ──────────────────────────────

def build_max_projection(image_path: str) -> tuple[np.ndarray, int]:
    """
    OR all Z-layers of the stack into one 2-D binary image (uint8 0/1): a
    pixel is foreground if it is foreground in any layer. Layers are read one
    at a time to keep memory use low; the OR runs on the GPU if CuPy works.
    Returns (projection, number of layers).
    """
    use_gpu = gpu_available()
    if use_gpu:
        import cupy as xp
    else:
        xp = np

    lazy, store = open_stack(image_path)
    try:
        n_layers = lazy.shape[0] if lazy.ndim == 3 else 1
        proj = None
        for i in range(n_layers):
            layer = xp.asarray(load_layer(lazy, i).astype(bool))
            proj = layer if proj is None else proj | layer
    finally:
        store.close()

    if use_gpu:
        result = xp.asnumpy(proj).astype(np.uint8)
        del proj
        xp.get_default_memory_pool().free_all_blocks()
    else:
        result = proj.astype(np.uint8)

    print(f"  Max projection backend: {'CuPy GPU' if use_gpu else 'NumPy CPU'}")
    return result, n_layers


def build_raw_graph(skeleton: np.ndarray, pixel_size_um: float) -> nx.Graph:
    """
    Build the raw graph from the full skeleton, one connected piece at a time.

    Small pieces go straight through sknw. Large pieces (the main network can
    have millions of skeleton pixels) are processed in overlapping tiles at
    full resolution. Node positions are in full-image pixel coordinates.
    """
    labeled, n_pieces = sk_label(skeleton, return_num=True)
    print(f"  {int(skeleton.sum()):,} skeleton px  |  {n_pieces:,} connected pieces")

    merged = nx.Graph()
    node_offset = 0
    pad = 2

    for region in regionprops(labeled):
        r0, c0, r1, c1 = region.bbox
        r0, c0 = max(0, r0 - pad), max(0, c0 - pad)
        r1, c1 = min(skeleton.shape[0], r1 + pad), min(skeleton.shape[1], c1 + pad)

        crop = skeleton[r0:r1, c0:c1].astype(np.uint8)
        n_px = int(crop.sum())
        if n_px == 0:
            continue

        if n_px >= config.TILE_COMPONENT_THRESH:
            # Already returns full-image coordinates.
            sub_g = build_tiled_graph(
                crop, r_origin=r0, c_origin=c0, pixel_size_um=pixel_size_um,
                tile_size=config.TILE_SIZE, overlap=config.TILE_OVERLAP,
                merge_radius=config.TILE_MERGE_RADIUS,
            )
        else:
            try:
                sub_g = build_graph(crop, pixel_size_um)
            except Exception as exc:
                print(f"  Skipping piece of {n_px:,} px: {exc}")
                continue
            for _, data in sub_g.nodes(data=True):
                if "o_row" in data:
                    data["o_row"] = float(data["o_row"]) + r0
                    data["o_col"] = float(data["o_col"]) + c0

        if sub_g.number_of_nodes() == 0:
            continue

        # Shift node IDs so pieces don't collide when merged.
        sub_g = nx.relabel_nodes(sub_g, {n: n + node_offset for n in sub_g.nodes()})
        merged.update(sub_g)
        node_offset += sub_g.number_of_nodes() + 1

    print(f"  Raw graph: {merged.number_of_nodes():,}N  {merged.number_of_edges():,}E")
    return merged


def _refine_params() -> dict:
    """Refinement settings from config.py, as keyword arguments for refine_graph()."""
    return dict(
        min_length_um=config.MIN_EDGE_LENGTH_UM,
        angle_threshold_deg=config.ANGULAR_THRESHOLD_DEG,
        min_component_nodes=config.MIN_COMPONENT_NODES,
        min_component_fraction=config.MIN_COMPONENT_FRACTION,
        dead_end_length_um=config.DEAD_END_LENGTH_UM,
        min_component_length_um=config.MIN_COMPONENT_LENGTH_UM,
        max_merge_distance_um=config.MERGE_NODE_DISTANCE_UM,
        junction_contraction_length_um=config.JUNCTION_CONTRACTION_LENGTH_UM,
        keep_largest_component_only=config.KEEP_ONLY_LARGEST_COMPONENT,
        component_rank_to_keep=config.COMPONENT_RANK_TO_KEEP,
        component_bridge_top_n=config.COMPONENT_BRIDGE_TOP_N,
        component_bridge_max_distance_um=config.COMPONENT_BRIDGE_MAX_DISTANCE_UM,
        component_bridge_max_angle_deg=config.COMPONENT_BRIDGE_MAX_ANGLE_DEG,
        pixel_size_um=config.PIXEL_SIZE_UM,
        component_analysis_k_sample=config.ANALYSIS_K_SAMPLE,
    )


# ── Rasterising graphs for figures ───────────────────────────────────────────

def _scaled_px(node_data: dict, factor: float, shape: tuple) -> tuple[int, int]:
    """Node position divided by `factor`, rounded and clipped to `shape`."""
    h, w = shape
    r = max(0, min(int(round(float(node_data["o_row"]) / factor)), h - 1))
    c = max(0, min(int(round(float(node_data["o_col"]) / factor)), w - 1))
    return r, c


def _node_mask(G: nx.Graph, shape: tuple, factor: int) -> np.ndarray:
    """Small dot at every node position."""
    pts = np.zeros(shape, dtype=np.uint8)
    for _, nd in G.nodes(data=True):
        if "o_row" in nd:
            pts[_scaled_px(nd, factor, shape)] = 1
    return dilation(pts, disk(1))


def _graph_on_skeleton_mask(G: nx.Graph, shape: tuple, factor: int,
                            skel_thumb: np.ndarray) -> np.ndarray:
    """
    Show which skeleton pixels are covered by the graph: every edge is drawn
    as a straight line, widened into a corridor, and intersected with the
    skeleton. Node dots are added on top.
    """
    corridor = np.zeros(shape, dtype=np.uint8)
    for u, v in G.edges():
        nu, nv = G.nodes[u], G.nodes[v]
        if "o_row" not in nu or "o_row" not in nv:
            continue
        rr, cc = draw_line(*_scaled_px(nu, factor, shape), *_scaled_px(nv, factor, shape))
        corridor[rr, cc] = 1
    corridor = dilation(corridor, disk(4))
    return np.logical_or(skel_thumb & corridor, _node_mask(G, shape, factor)).astype(np.uint8)


def _graph_mask(G: nx.Graph, image_shape: tuple, downsample: int) -> np.ndarray:
    """All edges and nodes of G as a slightly thickened mask, at 1/downsample scale."""
    shape = (image_shape[0] // downsample, image_shape[1] // downsample)
    mask = np.zeros(shape, dtype=np.uint8)
    for u, v in G.edges():
        nu, nv = G.nodes[u], G.nodes[v]
        if "o_row" in nu and "o_row" in nv:
            rr, cc = draw_line(*_scaled_px(nu, downsample, shape), *_scaled_px(nv, downsample, shape))
            mask[rr, cc] = 1
    for _, nd in G.nodes(data=True):
        if "o_row" in nd:
            rr, cc = draw_disk(_scaled_px(nd, downsample, shape), 1, shape=shape)
            mask[rr, cc] = 1
    return dilation(mask, disk(1))


def _overlay_rgb(base: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    rgb = np.stack([base, base, base], axis=-1).astype(float)
    rgb[mask > 0] = color
    return rgb


# ── Figures ──────────────────────────────────────────────────────────────────

YELLOW, ORANGE, GREEN = [1.0, 1.0, 0.0], [1.0, 0.55, 0.0], [0.2, 0.9, 0.3]


def save_two_panel_figure(tissue_thumb, overlay, color, title_left, title_right, path, dpi=400):
    """Overlay on the tissue mask (left) and the overlay alone on black (right)."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(26, 13), facecolor="black",
                                     gridspec_kw={"wspace": 0.03})
    ax_l.imshow(_overlay_rgb(tissue_thumb, overlay, color))
    ax_r.imshow(_overlay_rgb(np.zeros_like(tissue_thumb), overlay, color))
    for ax, title in ((ax_l, title_left), (ax_r, title_right)):
        ax.set_title(title, color="white", fontsize=14, pad=10)
        ax.axis("off")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def save_comparison_figure(panels, path, dpi=400):
    """
    One row of panels showing the pipeline stages side by side.
    `panels` is a list of (image, title); 2-D images are drawn in grey.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(12 * len(panels), 12),
                             facecolor="black", gridspec_kw={"wspace": 0.03})
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, color="white", fontsize=13, pad=10, fontfamily="monospace")
        ax.axis("off")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def save_timing_chart(times: dict, title: str, path: str):
    labels, values = list(times), list(times.values())
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c", "#9b59b6", "#1abc9c"]
    bars = ax.barh(labels, values, color=colors[:len(labels)], edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                _format_duration(val), va="center", fontsize=10)
    ax.set_xlabel("Wall-clock time (seconds)")
    ax.set_title(title, fontweight="bold")
    ax.set_facecolor("#f8f9fa")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Colours for the component preview, by component size rank. The last entry
# is reused for all smaller components.
COMPONENT_COLORS = [
    (0, 114, 189),    # blue: largest component
    (217, 118, 51),   # orange
    (32, 178, 81),    # green
    (162, 20, 132),   # purple
    (0, 163, 163),    # teal
    (177, 122, 162),  # lavender
    (220, 150, 0),    # amber
    (120, 120, 120),  # grey: everything smaller
]
DEAD_END_COLOR = (40, 40, 40)
BRIDGE_COLOR = (230, 25, 25)


def save_component_preview(G: nx.Graph, shape: tuple, path: str,
                           pixel_size_um: float, downsample: int = 4):
    """
    Draw the graph on white with each connected component in its own colour
    (largest = blue). Junctions (degree >= 3) are drawn as larger dots, dead
    ends as dark dots and synthetic bridge edges in red. Downsampled 4× by
    default so the file opens quickly in any image viewer.
    """
    h, w = shape[0] // downsample, shape[1] // downsample
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    color_of = {}
    for rank, comp in enumerate(sorted(nx.connected_components(G), key=len, reverse=True)):
        color = COMPONENT_COLORS[min(rank, len(COMPONENT_COLORS) - 1)]
        color_of.update(dict.fromkeys(comp, color))

    for u, v, edata in G.edges(data=True):
        nu, nv = G.nodes[u], G.nodes[v]
        if "o_row" not in nu or "o_row" not in nv:
            continue
        dist_px = np.hypot(nv["o_row"] - nu["o_row"], nv["o_col"] - nu["o_col"])
        if "length_um" in edata and is_phantom_edge(dist_px, edata["length_um"], pixel_size_um):
            continue
        rr, cc = draw_line(*_scaled_px(nu, downsample, (h, w)), *_scaled_px(nv, downsample, (h, w)))
        img[rr, cc] = BRIDGE_COLOR if edata.get("bridge") else color_of[u]

    for n, nd in G.nodes(data=True):
        if "o_row" not in nd:
            continue
        deg = G.degree(n)
        color = DEAD_END_COLOR if deg == 1 else color_of[n]
        radius = 2 if deg >= 3 else 1
        rr, cc = draw_disk(_scaled_px(nd, downsample, (h, w)), radius, shape=(h, w))
        img[rr, cc] = color

    tifffile.imwrite(path, img)


# ── Metrics ──────────────────────────────────────────────────────────────────

def _graph_summary(G: nx.Graph) -> dict:
    degrees = [d for _, d in G.degree()]
    lengths = [d["length_um"] for _, _, d in G.edges(data=True) if "length_um" in d]
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "total_network_length_mm": sum(lengths) / 1000,
        "mean_degree": float(np.mean(degrees)) if degrees else 0,
        "max_degree": int(max(degrees)) if degrees else 0,
        "n_branch_points": sum(1 for d in degrees if d >= 3),
        "n_dead_ends": sum(1 for d in degrees if d == 1),
    }


def _print_summary(m: dict):
    sk, raw, rg, ga = m["skeletonization"], m["graph_building"], m["refined_graph"], m["graph_analysis"]
    print("\n  Summary")
    print("  " + "-" * 60)
    print(f"  Skeleton            {sk['skeleton_pixels']:,} px "
          f"({sk['skeleton_density_pct']:.1f}% of foreground)")
    print(f"  Raw graph           {raw['n_nodes']:,} nodes, {raw['n_edges']:,} edges, "
          f"{raw['total_network_length_mm']:.1f} mm")
    print(f"  Refined graph       {rg['n_nodes']:,} nodes, {rg['n_edges']:,} edges, "
          f"{rg['total_network_length_mm']:.1f} mm "
          f"(-{m['refinement']['total_reduction_nodes_pct']}% nodes)")
    print(f"  Components          {rg['n_connected_components']:,}")
    print(f"  Branch points       {rg['n_branch_points']:,}   dead ends: {rg['n_dead_ends']:,}")
    print(f"  Degree              mean {rg['mean_degree']:.3f}, max {rg['max_degree']}")
    if "mean_edge_length_um" in ga:
        print(f"  Edge length         mean {ga['mean_edge_length_um']:.1f} um, "
              f"median {ga['median_edge_length_um']:.1f} um")
    print(f"  Junction density    {rg['junction_density_per_mm2']:.1f} /mm2")
    print(f"  Fibre density       {rg['fiber_length_mm_per_mm2']:.3f} mm/mm2")
    for comp in ga.get("components", []):
        line = f"  Component #{comp['rank']}        {comp['n_nodes']:,} nodes ({100 * comp['fraction']:.1f}%)"
        if comp["avg_shortest_path_um"] is not None:
            line += (f", avg shortest path {comp['avg_shortest_path_um']:.1f} um, "
                     f"mean betweenness {comp['mean_betweenness']:.6f}")
        print(line)
    print("  " + "-" * 60)


def write_summary_csv() -> str | None:
    """
    Collect metrics_maxproj.json of every processed image into one table,
    data/output/spin_summary.csv. Returns its path (None if nothing found).
    """
    rows = []
    for stem in sorted(os.listdir(config.OUTPUT_DIR)):
        json_path = os.path.join(config.OUTPUT_DIR, stem, "Metrics", "metrics_maxproj.json")
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path) as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        img = m.get("image", {})
        skel = m.get("skeletonization", {})
        raw = m.get("graph_building", {})
        rg = m.get("refined_graph", {})
        ga = m.get("graph_analysis", {})
        # If only one component was kept, the 2nd/3rd largest were measured
        # just before they were dropped; otherwise they are in graph_analysis.
        comps = rg.get("pre_prune_components") or ga.get("components", [])

        def comp(rank, key):
            return comps[rank - 1].get(key) if len(comps) >= rank else None

        rows.append({
            "image_stem":               stem,
            "area_mm2":                 img.get("area_mm2"),
            "n_z_layers":               img.get("n_z_layers"),
            "skeleton_density_pct":     skel.get("skeleton_density_pct"),
            "raw_n_nodes":              raw.get("n_nodes"),
            "raw_n_edges":              raw.get("n_edges"),
            "raw_length_mm":            raw.get("total_network_length_mm"),
            "n_nodes":                  rg.get("n_nodes"),
            "n_edges":                  rg.get("n_edges"),
            "n_connected_components":   rg.get("n_connected_components"),
            "n_branch_points":          rg.get("n_branch_points"),
            "n_dead_ends":              rg.get("n_dead_ends"),
            "mean_degree":              rg.get("mean_degree"),
            "max_degree":               rg.get("max_degree"),
            "total_network_length_mm":  rg.get("total_network_length_mm"),
            "junction_density_per_mm2": rg.get("junction_density_per_mm2"),
            "branch_density_per_mm2":   rg.get("branch_density_per_mm2"),
            "fiber_length_mm_per_mm2":  rg.get("fiber_length_mm_per_mm2"),
            "avg_shortest_path_um":     ga.get("avg_shortest_path_um"),
            "mean_betweenness":         ga.get("mean_betweenness"),
            "max_betweenness":          ga.get("max_betweenness"),
            "largest_cc_fraction":      ga.get("largest_cc_fraction"),
            "second_cc_nodes":          comp(2, "n_nodes"),
            "second_cc_fraction":       comp(2, "fraction"),
            "second_cc_avg_shortest_path_um": comp(2, "avg_shortest_path_um"),
            "third_cc_nodes":           comp(3, "n_nodes"),
            "third_cc_fraction":        comp(3, "fraction"),
            "third_cc_avg_shortest_path_um":  comp(3, "avg_shortest_path_um"),
        })

    if not rows:
        print("  No metrics found, summary CSV not written.")
        return None

    csv_path = os.path.join(config.OUTPUT_DIR, "spin_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Summary CSV ({len(rows)} image(s)) -> {csv_path}")
    return csv_path


# ── Full pipeline for one image ──────────────────────────────────────────────

def run_image(image_path: str):
    """Run all five steps on one image and write every output file."""
    timer = StepTimer()
    dirs, stem = _image_output_dirs(image_path)
    px_um = config.PIXEL_SIZE_UM

    # 1. Max projection
    print("[1/5] Building max projection ...")
    timer.start("1. Max projection")
    binary, n_layers = build_max_projection(image_path)
    shape = H, W = binary.shape
    fg_px = int(binary.sum())
    print(f"  Shape: {shape}  Foreground: {fg_px:,} px ({100 * fg_px / binary.size:.1f}%)  "
          f"[{timer.stop():.1f}s]")

    # Figures use a 2x downsampled copy (for typical ~6600 px images), which
    # still shows individual 1-px conduits.
    factor = max(1, max(shape) // 3000)
    tissue_thumb = (binary[::factor, ::factor] > 0).astype(np.uint8)
    thumb_shape = tissue_thumb.shape

    # 2. Skeletonisation
    print("\n[2/5] Skeletonising ...")
    timer.start("2. Skeletonise")
    skeleton = compute_skeleton(binary, save_path=os.path.join(dirs["refined"], "skeleton_maxproj.tif"))
    skeleton_px = int(skeleton.sum())
    skel_thumb_raw = skeleton[::factor, ::factor].astype(np.uint8)
    skel_thumb = dilation(skel_thumb_raw, disk(1))
    print(f"  [{timer.stop():.1f}s]")

    save_two_panel_figure(
        tissue_thumb, skel_thumb, YELLOW,
        f"Skeleton on tissue\n{skeleton_px:,} px  |  {px_um} um/px",
        f"Skeleton only\n{skeleton_px:,} px  |  1-px-wide centre lines",
        os.path.join(dirs["refined"], "skeleton_figure.tif"),
    )
    del binary
    gc.collect()

    # 3. Raw graph
    print("\n[3/5] Building raw graph ...")
    timer.start("3. Graph build")
    graph_raw = build_raw_graph(skeleton, px_um)
    print(f"  [{timer.stop():.1f}s]")
    del skeleton
    gc.collect()

    save_graphml(graph_raw, os.path.join(dirs["refined"], "reticular_graph_raw_maxproj.graphml"))
    raw = _graph_summary(graph_raw)
    raw_thumb = np.logical_or(skel_thumb, _node_mask(graph_raw, thumb_shape, factor)).astype(np.uint8)
    save_two_panel_figure(
        tissue_thumb, raw_thumb, ORANGE,
        f"Raw graph on tissue\n{raw['n_nodes']:,} N  {raw['n_edges']:,} E  |  "
        f"{raw['total_network_length_mm']:.1f} mm",
        f"Raw graph only\n{raw['n_nodes']:,} N  {raw['n_edges']:,} E",
        os.path.join(dirs["refined"], "raw_graph_figure.tif"),
    )

    # 4. Refinement
    print("\n[4/5] Refining graph ...")
    timer.start("4. Graph refine")
    graph, refine_steps, graph_pre_prune = refine_graph(graph_raw, **_refine_params())
    print(f"  [{timer.stop():.1f}s]")
    del graph_raw
    gc.collect()

    save_graphml(graph, os.path.join(dirs["refined"], "reticular_graph_maxproj.graphml"))
    tifffile.imwrite(os.path.join(dirs["refined"], "refined_skeleton_maxproj.tif"),
                     render_graph_as_skeleton(graph, shape, pixel_size_um=px_um) * 255)
    # Colour by component on the graph *before* the final component
    # selection; afterwards there may be only one component left to show.
    save_component_preview(graph_pre_prune if graph_pre_prune is not None else graph, shape,
                           os.path.join(dirs["refined"], "component_preview.tif"), px_um)

    # 5. Analysis, metrics and overview figure
    print("\n[5/5] Analysing graph and saving figures ...")
    timer.start("5. Analyse + figures")
    graph_metrics = analyse(graph, image_shape=shape, pixel_size_um=px_um,
                            k_sample=config.ANALYSIS_K_SAMPLE)

    ref = _graph_summary(graph)
    area_mm2 = H * W * px_um ** 2 / 1e6
    metrics = {
        "image": {
            "path": image_path,
            "stem": stem,
            "shape_px": [H, W],
            "n_z_layers": n_layers,
            "pixel_size_um": px_um,
            "area_mm2": round(area_mm2, 4),
        },
        "skeletonization": {
            "skeleton_pixels": skeleton_px,
            "foreground_pixels": fg_px,
            "skeleton_density_pct": round(100 * skeleton_px / max(fg_px, 1), 2),
        },
        "graph_building": {
            "n_nodes": raw["n_nodes"],
            "n_edges": raw["n_edges"],
            "total_network_length_mm": round(raw["total_network_length_mm"], 3),
            "mean_degree": round(raw["mean_degree"], 4),
            "n_branch_points": raw["n_branch_points"],
            "n_dead_ends": raw["n_dead_ends"],
        },
        "refinement": {
            "passes": refine_steps,
            "total_reduction_nodes_pct": round(100 * (1 - ref["n_nodes"] / max(raw["n_nodes"], 1)), 1),
            "total_reduction_edges_pct": round(100 * (1 - ref["n_edges"] / max(raw["n_edges"], 1)), 1),
        },
        "refined_graph": {
            "n_nodes": ref["n_nodes"],
            "n_edges": ref["n_edges"],
            "n_connected_components": nx.number_connected_components(graph),
            "n_branch_points": ref["n_branch_points"],
            "n_dead_ends": ref["n_dead_ends"],
            "mean_degree": round(ref["mean_degree"], 4),
            "max_degree": ref["max_degree"],
            "total_network_length_mm": round(ref["total_network_length_mm"], 3),
            "junction_density_per_mm2": round(ref["n_nodes"] / area_mm2, 2),
            "branch_density_per_mm2": round(ref["n_branch_points"] / area_mm2, 2),
            "fiber_length_mm_per_mm2": round(ref["total_network_length_mm"] / area_mm2, 4),
            # Top-3 components measured just before the final component
            # selection (None if all components are kept).
            "pre_prune_components": refine_steps[-1]["pre_prune_top_components"],
        },
        "graph_analysis": graph_metrics,
    }
    _print_summary(metrics)

    refined_thumb = _graph_on_skeleton_mask(graph, thumb_shape, factor, skel_thumb_raw)
    black = np.zeros_like(tissue_thumb)
    rg = metrics["refined_graph"]
    save_comparison_figure([
        (tissue_thumb,
         f"Step 1  Binary input\n{n_layers} Z-layers    {W}x{H} px    {px_um} um/px"),
        (_overlay_rgb(black, skel_thumb, YELLOW),
         f"Step 2  Skeleton\n{skeleton_px:,} px"),
        (_overlay_rgb(black, raw_thumb, ORANGE),
         f"Step 3  Graph building\n{raw['n_nodes']:,} N    {raw['n_edges']:,} E    "
         f"{raw['total_network_length_mm']:.0f} mm"),
        (_overlay_rgb(black, refined_thumb, GREEN),
         f"Step 4  Refined graph\n{rg['n_nodes']:,} N    {rg['n_edges']:,} E    "
         f"{rg['total_network_length_mm']:.0f} mm    {rg['n_connected_components']} comp    "
         f"{rg['n_branch_points']:,} junctions"),
    ], os.path.join(dirs["comparison"], "spin_analysis_maxproj.tif"))
    print(f"  [{timer.stop():.1f}s]")

    timer.report()
    metrics["timing_s"] = {k: round(v, 3) for k, v in timer.times.items()}
    with open(os.path.join(dirs["metrics"], "metrics_maxproj.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    save_timing_chart(timer.times, f"Per-step timing: {stem}",
                      os.path.join(dirs["comparison"], "timing_maxproj.tif"))

    print(f"  All outputs written to {dirs['base']}")


def run_image_with_log(image_path: str):
    """run_image(), with everything it prints also saved to Metrics/run_log.txt."""
    dirs, _ = _image_output_dirs(image_path)
    with open(os.path.join(dirs["metrics"], "run_log.txt"), "w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)):
            run_image(image_path)


def run_all():
    """Process every image in data/Input/, then write the summary CSV."""
    images = _list_input_images()
    if not images:
        print(f"No TIFF images found in {config.INPUT_DIR}")
        return

    print(f"Found {len(images)} image(s) to process")
    t_start = time.perf_counter()
    failed = []

    for i, path in enumerate(images, 1):
        print(f"\n{'=' * 65}\n[{i}/{len(images)}] {os.path.basename(path)}\n{'=' * 65}")
        try:
            run_image_with_log(path)
        except Exception as exc:
            # Keep going with the remaining images; report failures at the end.
            print(f"ERROR: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failed.append((os.path.basename(path), str(exc)))
        finally:
            plt.close("all")
            gc.collect()

    print(f"\n{'=' * 65}")
    print(f"Done: {len(images) - len(failed)}/{len(images)} succeeded "
          f"({(time.perf_counter() - t_start) / 60:.1f} min)")
    for name, err in failed:
        print(f"  FAILED  {name}: {err}")
    print(f"{'=' * 65}\n")

    write_summary_csv()


# ── Refinement diagnostics ───────────────────────────────────────────────────

def _render_graph(G, image_shape, path, color, downsample):
    """Graph in a single colour on white."""
    mask = _graph_mask(G, image_shape, downsample)
    img = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    img[mask > 0] = color
    tifffile.imwrite(path, img)


def _render_diff(G_before, G_after, image_shape, path, downsample):
    """
    What a step removed: G_before is drawn in red, then G_after in green on
    top, so only the parts that disappeared stay red. Works purely on pixel
    positions, so it doesn't matter if node IDs changed in between.
    """
    before = _graph_mask(G_before, image_shape, downsample)
    after = _graph_mask(G_after, image_shape, downsample)
    img = np.full((*before.shape, 3), 255, dtype=np.uint8)
    img[before > 0] = (230, 25, 25)
    img[after > 0] = (30, 160, 60)
    tifffile.imwrite(path, img)


def run_debug_passes(image_path: str):
    """
    Run the pipeline on one image and save an image of the graph after each
    refinement pass, plus a diff showing what that pass removed. Meant for
    checking and tuning the refinement thresholds.

    Output in data/output/<image>/Refinement Passes/:
        dropped_skeletonization.tif     skeleton (yellow) on the tissue mask (grey)
        dropped_graph_construction.tif  skeleton pixels not covered by the raw graph
                                        (red); should be close to empty
        00_raw_graph.tif                raw graph
        NN_<pass>_after.tif             graph after pass NN
        NN_<pass>_dropped.tif           green = kept, red = removed by pass NN
        dropped_refinement.tif          raw graph vs. final refined graph
    """
    dirs, stem = _image_output_dirs(image_path)
    out_dir = os.path.join(dirs["base"], "Refinement Passes")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/4] Building max projection ({stem}) ...")
    binary, _ = build_max_projection(image_path)
    shape = binary.shape
    downsample = max(1, max(shape) // 1650)

    print("\n[2/4] Skeletonising ...")
    skeleton = compute_skeleton(binary)

    print("\n[3/4] Building raw graph ...")
    graph_raw = build_raw_graph(skeleton, config.PIXEL_SIZE_UM)

    fg_thumb = binary[::downsample, ::downsample] > 0
    skel_thumb = skeleton[::downsample, ::downsample] > 0
    del binary, skeleton
    gc.collect()

    img = np.full((*fg_thumb.shape, 3), 255, dtype=np.uint8)
    img[fg_thumb & ~skel_thumb] = (200, 200, 200)
    img[skel_thumb] = (240, 190, 0)
    tifffile.imwrite(os.path.join(out_dir, "dropped_skeletonization.tif"), img)

    covered = _graph_on_skeleton_mask(graph_raw, skel_thumb.shape, downsample,
                                      skel_thumb.astype(np.uint8))
    img = np.full((*fg_thumb.shape, 3), 255, dtype=np.uint8)
    img[skel_thumb & (covered == 0)] = (200, 0, 0)
    img[covered > 0] = (30, 160, 60)
    tifffile.imwrite(os.path.join(out_dir, "dropped_graph_construction.tif"), img)

    _render_graph(graph_raw, shape, os.path.join(out_dir, "00_raw_graph.tif"),
                  color=(255, 140, 0), downsample=downsample)

    print("\n[4/4] Refining graph (one image per pass) ...")

    def on_pass(number, name, G_before, G_after):
        tag = f"{number:02d}_{name}"
        _render_graph(G_after, shape, os.path.join(out_dir, f"{tag}_after.tif"),
                      color=(30, 160, 60), downsample=downsample)
        _render_diff(G_before, G_after, shape, os.path.join(out_dir, f"{tag}_dropped.tif"),
                     downsample=downsample)

    graph_refined, _, _ = refine_graph(graph_raw, on_pass=on_pass, **_refine_params())
    _render_diff(graph_raw, graph_refined, shape,
                 os.path.join(out_dir, "dropped_refinement.tif"), downsample=downsample)

    print(f"\nPass images written to {out_dir}")


# ── Command line ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SPIN: graph extraction and analysis of lymph node reticular networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python pipeline.py --all\n"
            "  python pipeline.py --image data/Input/sample.ome.tif\n"
            "  python pipeline.py --image data/Input/sample.ome.tif --debug-passes\n"
        ),
    )
    parser.add_argument("--all", action="store_true",
                        help="process every TIFF in data/Input/")
    parser.add_argument("--image", metavar="PATH",
                        help="process a single image")
    parser.add_argument("--debug-passes", action="store_true",
                        help="with --image: also save a before/after image for every "
                             "refinement pass (to 'Refinement Passes/')")
    args = parser.parse_args()

    if args.all == bool(args.image):
        parser.error("use either --all or --image PATH")
    if args.debug_passes and not args.image:
        parser.error("--debug-passes requires --image PATH")

    print("SPIN pipeline")
    print(f"Max projection : {'CuPy (GPU)' if gpu_available() else 'NumPy (CPU)'}")
    print(f"Skeletoniser   : {'cuCIM (GPU)' if cucim_available() else 'scikit-image (CPU)'}")
    print("=" * 60)

    if args.all:
        run_all()
    elif args.debug_passes:
        run_debug_passes(args.image)
    else:
        run_image_with_log(args.image)


if __name__ == "__main__":
    main()
