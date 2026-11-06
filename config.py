"""
All pipeline settings in one place.

Thresholds are given in micrometres and converted to pixels internally
using PIXEL_SIZE_UM.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# Input stacks go in data/Input/, results are written to data/output/.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "Input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

# ── Physical scale ────────────────────────────────────────────────────────────
# The TIFF files carry no calibration metadata, so this value has to be set
# by hand. Every µm-based threshold and metric depends on it.
PIXEL_SIZE_UM = 0.6

# ── Graph construction ────────────────────────────────────────────────────────
# Skeleton pieces larger than TILE_COMPONENT_THRESH pixels are processed in
# overlapping tiles (see spin/graph_builder.py); smaller pieces in one go.
# Duplicate junctions from the tile overlaps are merged within
# TILE_MERGE_RADIUS pixels.
TILE_COMPONENT_THRESH = 200_000  # px
TILE_SIZE = 2048                 # px
TILE_OVERLAP = 256               # px
TILE_MERGE_RADIUS = 5.0          # px

# ── Graph refinement (see spin/graph_refiner.py for the ten passes) ─────────

# Pass 2: merge nodes closer than this (duplicate junctions).
MERGE_NODE_DISTANCE_UM = 2.0

# Pass 3: remove dead-end stubs shorter than this. These are sub-pixel
# skeletonisation artefacts (3.6 µm = 6 voxels before downsampling).
MIN_EDGE_LENGTH_UM = 3.6

# Pass 4: remove pass-through nodes whose angle is within this many degrees
# of a straight line. Nodes at bends sharper than that are kept.
ANGULAR_THRESHOLD_DEG = 30

# Pass 5: contract edges shorter than this when both ends are junctions or
# pass-through nodes. One thick branch point is often skeletonised into a
# small cluster of nodes (typically a triangle); tip pruning can't remove
# those because none of the nodes is a dead end. The value matches the
# typical junction-to-junction spacing (~4 µm) we measured on this dataset.
JUNCTION_CONTRACTION_LENGTH_UM = 4.0

# Pass 6: after pass 4 has merged edges into whole branches, remove dead-end
# branches shorter than this. Follows the "~3× vessel diameter" rule used by
# Blinder et al. (2013, Nat Neurosci 16:889) for vascular networks.
DEAD_END_LENGTH_UM = 15.0

# Pass 7: join the COMPONENT_BRIDGE_TOP_N largest components with a
# synthetic edge (tagged bridge=True) when their closest, mutually aligned
# nodes are at most COMPONENT_BRIDGE_MAX_DISTANCE_UM apart. "Aligned" means
# both end points point towards each other within
# COMPONENT_BRIDGE_MAX_ANGLE_DEG. Set the distance to 0 to switch it off.
COMPONENT_BRIDGE_MAX_DISTANCE_UM = 30.0
COMPONENT_BRIDGE_TOP_N = 3
COMPONENT_BRIDGE_MAX_ANGLE_DEG = 45.0

# Pass 8: remove components with fewer nodes than
#   max(MIN_COMPONENT_NODES, MIN_COMPONENT_FRACTION × nodes in the largest).
MIN_COMPONENT_NODES = 20
MIN_COMPONENT_FRACTION = 0.005

# Pass 9: remove components whose total edge length is below this; they are
# too small to form a meaningful transport path (Kelch et al. 2015,
# eLife 4:e06493).
MIN_COMPONENT_LENGTH_UM = 50.0

# Pass 10: keep only selected components, ranked by node count
# (1 = largest). Either one rank, e.g. 1, or a list such as [1, 2, 3] to
# keep several components side by side. Set KEEP_ONLY_LARGEST_COMPONENT to
# False to keep everything that survived the filters above.
KEEP_ONLY_LARGEST_COMPONENT = True
COMPONENT_RANK_TO_KEEP = 1

# ── Graph analysis ────────────────────────────────────────────────────────────
# Number of randomly sampled source nodes for the average shortest path and
# betweenness estimates. More samples are more accurate but slower.
ANALYSIS_K_SAMPLE = 200
