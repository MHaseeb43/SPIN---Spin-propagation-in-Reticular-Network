# SPIN – Signal Propagation in the Reticular Network of Human Lymph Nodes

Lymph nodes are held together by a mesh of fibroblastic reticular cells that
form tiny channels, called **conduits**. These conduits carry small molecules,
antigens and chemokines through the node, mostly by passive transport, so the
shape of the network affects how well signals can spread.

SPIN measures that shape. It takes binarised confocal Z-stacks of the conduit
network, converts the network into a graph (junctions = nodes, conduit
segments = edges), removes imaging and skeletonisation artefacts, and
computes graph metrics that can serve as simple proxies for signal
propagation.

## Pipeline

```
binary Z-stack ─► max projection ─► skeleton ─► raw graph ─► refined graph ─► metrics
```

| Step | Module | What happens |
|---|---|---|
| 1. Max projection | `pipeline.py` | All Z-layers are OR-ed into one 2-D binary image. |
| 2. Skeletonisation | `spin/skeletoniser.py` | The network is thinned to 1-px-wide centre lines (Lee 1994). |
| 3. Graph building | `spin/graph_builder.py` | `sknw` turns the skeleton into a graph. Large pieces are processed in overlapping tiles at full resolution. |
| 4. Refinement | `spin/graph_refiner.py` | Ten passes that remove artefacts (see below). |
| 5. Analysis | `spin/analyser.py` | Degree, length and density statistics, average shortest path and betweenness centrality. |

### Refinement passes

| # | Pass | Removes | Default |
|---|---|---|---|
| 1 | Phantom edges | sknw tile-stitching edges whose endpoints are much further apart than their path length | distance > 3 × path + 5 px |
| 2 | Merge close nodes | Duplicate junctions | ≤ 2 µm |
| 3 | Tip pruning | Very short dead-end stubs | < 3.6 µm |
| 4 | Degree-2 dissolve | Pass-through nodes on nearly straight paths | within 30° of straight |
| 5 | Junction contraction | Tiny loops/triangles at thick branch points | < 4 µm |
| 6 | Dead-end pruning | Short dead-end branches (after merging edges) | < 15 µm |
| 7 | Component bridging | Joins the 3 largest components when aligned end points are close (synthetic edge, `bridge=True`) | ≤ 30 µm, ≤ 45° |
| 8 | Node-count filter | Small fragments | < max(20, 0.5 % of largest) nodes |
| 9 | Length filter | Fragments with little total length | < 50 µm |
| 10 | Component selection | Everything except the chosen component(s) | keep largest |

All thresholds are set in [`config.py`](config.py), with a short explanation
for each one.

## Installation

Python 3.11 was used for development.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

GPU support is optional. CuPy speeds up the max projection and cuCIM the
skeletonisation (cuCIM is only available via conda on Linux). Without them
everything runs on the CPU.

## Usage

Put the binarised `.ome.tif` stacks in `data/Input/` (foreground = conduit,
background = 0), then run:

```bash
python pipeline.py --all                       # all images in data/Input/
python pipeline.py --image data/Input/x.ome.tif   # one image
python pipeline.py --image data/Input/x.ome.tif --debug-passes
```

`--debug-passes` also saves an image of the graph after every refinement
pass, plus a red/green diff of what that pass removed. This is useful when
tuning the thresholds.

Set `PIXEL_SIZE_UM` in `config.py` to match your data (0.6 µm/px for our
images). The TIFF files carry no calibration, and every µm-based threshold
and metric depends on this value.

## Output

```
data/output/<image name>/
├── Refined Graph/
│   ├── skeleton_maxproj.tif                 skeleton
│   ├── skeleton_figure.tif                  skeleton on tissue / on black
│   ├── reticular_graph_raw_maxproj.graphml  raw graph
│   ├── raw_graph_figure.tif                 raw graph on tissue / on black
│   ├── reticular_graph_maxproj.graphml      refined graph
│   ├── refined_skeleton_maxproj.tif         refined graph drawn as a skeleton
│   └── component_preview.tif                components coloured by size rank
├── Metrics/
│   ├── metrics_maxproj.json                 all measurements, per-pass counts
│   └── run_log.txt                          full console output
└── Comparison/
    ├── spin_analysis_maxproj.tif            4-panel overview of the stages
    └── timing_maxproj.tif                   runtime per step

data/output/spin_summary.csv                 one row per image (after --all)
```

Graphs are saved as GraphML and can be opened with NetworkX, Gephi or
Cytoscape. Node positions are in the `o_row` / `o_col` attributes (pixels),
and edge lengths in `length_um`.

## Metrics

| Metric | Meaning |
|---|---|
| Nodes, edges, degree distribution | Size and connectivity of the network |
| Branch points / dead ends | Nodes with degree ≥ 3 / degree 1 |
| Edge length statistics, total length | Segment lengths in µm, total network length in mm |
| Junction and fibre density | Junctions per mm², mm of conduit per mm² |
| Connected components | Number and size of separate sub-networks |
| Average shortest path | Mean distance along the conduits between two nodes (sampled) |
| Betweenness centrality | How many shortest paths pass through a node; highlights bottlenecks (sampled) |

The two path-based metrics are estimated from `ANALYSIS_K_SAMPLE` randomly
chosen source nodes. They therefore vary slightly between runs.

## Project layout

```
pipeline.py          command line entry point, runs all steps
config.py            all settings
spin/
  loader.py          lazy OME-TIFF reading (tifffile + zarr + dask)
  skeletoniser.py    skeletonisation (scikit-image or cuCIM)
  graph_builder.py   skeleton → graph (sknw, tiled for large pieces)
  graph_refiner.py   the ten refinement passes
  analyser.py        graph metrics
  utils.py           GraphML export, drawing helpers
  gpu_utils.py       optional GPU detection with CPU fallback
```

## References

- Lee, Kashyap & Chu (1994). Building skeleton models via 3-D medial
  surface/axis thinning algorithms. *CVGIP: Graphical Models and Image
  Processing* 56(6), 462–478.
- Blinder et al. (2013). The cortical angiome: an interconnected vascular
  network with noncolumnar patterns of blood flow. *Nature Neuroscience*
  16, 889–897.
- Kelch et al. (2015). *eLife* 4, e06493.
