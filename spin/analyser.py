"""
Graph metrics for the refined reticular network.

Besides basic size, degree and length statistics, we compute two measures
that serve as simple proxies for how well signals could travel through the
network:

  * average shortest path (µm): how far apart, along the conduits, two
    random points of the network are
  * betweenness centrality: which junctions lie on many shortest paths,
    i.e. potential bottlenecks

Both are expensive on graphs with tens of thousands of nodes, so they are
estimated from `k_sample` randomly chosen source nodes.
"""

import random
import numpy as np
import networkx as nx


def analyse_component(G_sub: nx.Graph, total_n: int, k_sample: int, rank: int) -> dict:
    """Size, shortest-path and betweenness statistics for one connected component."""
    n_sub = G_sub.number_of_nodes()
    comp = {
        "rank": rank,
        "n_nodes": n_sub,
        "n_edges": G_sub.number_of_edges(),
        "fraction": n_sub / total_n,
        "avg_shortest_path_um": None,
        "mean_betweenness": None,
        "max_betweenness": None,
        "top10_betweenness": [],
    }
    if n_sub < 2:
        return comp

    k = min(k_sample, n_sub)

    # Average shortest path: run Dijkstra from k random sources and average
    # over all reachable targets.
    print(f"  Average shortest path on component #{rank} ({n_sub:,} nodes, k={k}) ...")
    total_dist, pair_count = 0.0, 0
    for src in random.sample(list(G_sub.nodes()), k):
        dists = nx.single_source_dijkstra_path_length(G_sub, src, weight="length_um")
        total_dist += sum(dists.values())     # distance to src itself is 0
        pair_count += len(dists) - 1
    comp["avg_shortest_path_um"] = total_dist / pair_count if pair_count else None

    print(f"  Betweenness centrality on component #{rank} (k={k} pivots) ...")
    centrality = nx.betweenness_centrality(G_sub, k=k, weight="length_um", normalized=True)
    values = list(centrality.values())
    comp["mean_betweenness"] = float(np.mean(values))
    comp["max_betweenness"] = float(max(values))
    comp["top10_betweenness"] = [
        {"node": int(node), "centrality": float(c)}
        for node, c in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    return comp


def analyse(G: nx.Graph, image_shape=None, pixel_size_um: float | None = None,
            k_sample: int = 100) -> dict:
    """
    Compute all metrics for the refined graph.

    Parameters
    ----------
    G             : refined graph (simple nx.Graph)
    image_shape   : (height, width) in pixels, used for density metrics
    pixel_size_um : µm per pixel, used for density metrics
    k_sample      : number of sampled sources for shortest path / betweenness
    """
    n, e = G.number_of_nodes(), G.number_of_edges()
    metrics = {"n_nodes": n, "n_edges": e}
    if n == 0:
        print("  Graph is empty, nothing to analyse.")
        return metrics

    # Degree statistics
    degrees = [d for _, d in G.degree()]
    metrics["mean_degree"] = float(np.mean(degrees))
    metrics["median_degree"] = float(np.median(degrees))
    metrics["max_degree"] = int(max(degrees))
    vals, counts = np.unique(degrees, return_counts=True)
    metrics["degree_distribution"] = {int(k): int(v) for k, v in zip(vals, counts)}
    metrics["n_branch_points"] = sum(1 for d in degrees if d >= 3)
    metrics["n_dead_ends"] = sum(1 for d in degrees if d == 1)

    # Edge length statistics
    lengths = np.array([d["length_um"] for _, _, d in G.edges(data=True) if "length_um" in d])
    if lengths.size:
        metrics["mean_edge_length_um"] = float(lengths.mean())
        metrics["median_edge_length_um"] = float(np.median(lengths))
        metrics["std_edge_length_um"] = float(lengths.std())
        metrics["min_edge_length_um"] = float(lengths.min())
        metrics["max_edge_length_um"] = float(lengths.max())
        for p in (10, 25, 75, 90, 95):
            metrics[f"edge_length_p{p}_um"] = float(np.percentile(lengths, p))
    metrics["total_network_length_um"] = float(lengths.sum())
    metrics["total_network_length_mm"] = metrics["total_network_length_um"] / 1000.0

    # Densities relative to the imaged area
    if image_shape is not None and pixel_size_um is not None:
        h_px, w_px = image_shape
        area_mm2 = h_px * w_px * pixel_size_um ** 2 / 1e6
        metrics["image_area_mm2"] = float(area_mm2)
        metrics["junction_density_per_mm2"] = float(n / area_mm2)
        metrics["branch_point_density_per_mm2"] = float(metrics["n_branch_points"] / area_mm2)
        metrics["fiber_length_density_mm_per_mm2"] = float(metrics["total_network_length_mm"] / area_mm2)

    # Path-based metrics for the three largest components
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    metrics["n_connected_components"] = len(components)
    metrics["components"] = [
        analyse_component(G.subgraph(components[i]).copy(), n, k_sample, rank=i + 1)
        for i in range(min(3, len(components)))
    ]

    # Shortcuts to the largest component (used by the summary CSV)
    top = metrics["components"][0]
    metrics["largest_cc_nodes"] = top["n_nodes"]
    metrics["largest_cc_edges"] = top["n_edges"]
    metrics["largest_cc_fraction"] = top["fraction"]
    metrics["avg_shortest_path_um"] = top["avg_shortest_path_um"]
    metrics["mean_betweenness"] = top["mean_betweenness"]
    metrics["max_betweenness"] = top["max_betweenness"]
    metrics["top10_betweenness"] = top["top10_betweenness"]

    return metrics
