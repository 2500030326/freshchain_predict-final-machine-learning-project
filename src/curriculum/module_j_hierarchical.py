"""
Module J: Hierarchical Agglomerative Clustering
Reference Deck: M4-Session2-Hierarchical_Clustering-new.pptx
Authoritative requirements:
- Agglomerative bottom-up clustering mechanics & Lance-Williams update
- 4 Linkages: Single, Complete, Average, Ward
- Exact 5-Point Running Dataset: A=2, B=4, C=10, D=12, E=18
- Exact Pairwise Distance Matrix from Slide 6
- Exact Ward Linkage Merge Sequence from Slide 10 & 13:
    (A,B) -> h=1.41
    (C,D) -> h=1.41
    (CD,E) -> h=5.72
    (AB,CDE) -> h=11.32 (Root)
- Dendrogram Cut & Optimal Gap Heuristic:
    Largest gap between 5.72 and 11.32 (gap = 5.60) -> Optimal cut at h=8 gives K=2
- Slide 19 Retail Customer Segmentation Case Study (6 customers: P1-P6)
- Viva-Voce Questions V1-V6
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform


def get_5point_dataset() -> tuple[np.ndarray, list[str]]:
    """5 points in R^1 from Slide 6: A=2, B=4, C=10, D=12, E=18."""
    X = np.array([2.0, 4.0, 10.0, 12.0, 18.0]).reshape(-1, 1)
    labels = ["A", "B", "C", "D", "E"]
    return X, labels


def compute_5point_distance_matrix() -> pd.DataFrame:
    """Pairwise Euclidean distance matrix from Slide 6."""
    X, labels = get_5point_dataset()
    dist_mat = squareform(pdist(X, metric="euclidean"))
    return pd.DataFrame(dist_mat, index=labels, columns=labels)


def solve_ward_merge_sequence() -> dict:
    """Ward merge sequence matching Slide 10 & 13."""
    X, labels = get_5point_dataset()
    Z_ward = linkage(X, method="ward")
    # Z matrix: [idx1, idx2, height, cluster_size]

    merges = []
    # Index mapping: 0:A, 1:B, 2:C, 3:D, 4:E, 5:v1, 6:v2, 7:v3, 8:v4
    node_names = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    for step, row in enumerate(Z_ward, 1):
        idx1, idx2, h, sz = int(row[0]), int(row[1]), float(row[2]), int(row[3])
        merged_name = f"({node_names[idx1]},{node_names[idx2]})"
        new_node_id = 4 + step
        node_names[new_node_id] = merged_name
        h_slide = h / np.sqrt(2.0)
        merges.append({
            "step": step,
            "join": merged_name,
            "height": round(h_slide, 2),
            "height_scipy": round(h, 2),
            "delta_j": round((h_slide ** 2), 2),
            "cluster_size": sz
        })

    # Cut heights table from Slide 13
    cut_table = [
        {"cut_range": "t > 11.32", "K": 1, "partition": "{A, B, C, D, E}"},
        {"cut_range": "5.72 < t <= 11.32", "K": 2, "partition": "{A, B}, {C, D, E} (Optimal - Largest Gap 5.60)"},
        {"cut_range": "1.41 < t <= 5.72", "K": 3, "partition": "{A, B}, {C, D}, {E}"},
        {"cut_range": "t <= 1.41", "K": 5, "partition": "All singletons: {A}, {B}, {C}, {D}, {E}"}
    ]

    return {
        "merge_sequence": merges,
        "cut_table": cut_table,
        "slide_reported_merges": [
            {"join": "(A,B)", "height": 1.41},
            {"join": "(C,D)", "height": 1.41},
            {"join": "(CD,E)", "height": 5.72},
            {"join": "(AB,CDE)", "height": 11.32}
        ]
    }


def solve_retail_case_study() -> dict:
    """Slide 19: 6 Retail Customers Segmentation Case Study."""
    # P1(10,2), P2(12,3), P3(11,2), P4(60,15), P5(62,14), P6(58,16)
    pts = np.array([
        [10.0, 2.0],
        [12.0, 3.0],
        [11.0, 2.0],
        [60.0, 15.0],
        [62.0, 14.0],
        [58.0, 16.0]
    ])
    labels = ["P1", "P2", "P3", "P4", "P5", "P6"]
    dist_df = pd.DataFrame(squareform(pdist(pts)), index=labels, columns=labels)
    Z = linkage(pts, method="ward")
    clusters_k2 = fcluster(Z, t=2, criterion="maxclust")

    cluster_assignment = dict(zip(labels, [int(c) for c in clusters_k2]))
    return {
        "customers": [
            {"id": "P1", "annual_spend_k": 10, "visits_pm": 2, "segment": cluster_assignment["P1"]},
            {"id": "P2", "annual_spend_k": 12, "visits_pm": 3, "segment": cluster_assignment["P2"]},
            {"id": "P3", "annual_spend_k": 11, "visits_pm": 2, "segment": cluster_assignment["P3"]},
            {"id": "P4", "annual_spend_k": 60, "visits_pm": 15, "segment": cluster_assignment["P4"]},
            {"id": "P5", "annual_spend_k": 62, "visits_pm": 14, "segment": cluster_assignment["P5"]},
            {"id": "P6", "annual_spend_k": 58, "visits_pm": 16, "segment": cluster_assignment["P6"]},
        ],
        "segments": {
            "Segment 1 (Value/Casual)": ["P1", "P2", "P3"],
            "Segment 2 (High-Value VIPs)": ["P4", "P5", "P6"]
        },
        "recommendation": "Optimal cut suggests K=2. Massive gap between low-spend cluster (Spend~11k, Visits~2) and VIP cluster (Spend~60k, Visits~15)."
    }


def run_module_j_walkthrough() -> dict:
    dist_df = compute_5point_distance_matrix()
    ward_res = solve_ward_merge_sequence()
    retail_res = solve_retail_case_study()

    return {
        "distance_matrix": dist_df.to_dict(),
        "ward_merges": ward_res,
        "retail_case_study": retail_res,
        "linkage_comparison": {
            "Single": {"Formula": "min pairwise dist", "Shape": "Chained, elongated", "Sensitivity": "Low to outliers, suffers from chaining"},
            "Complete": {"Formula": "max pairwise dist", "Shape": "Compact, spherical", "Sensitivity": "High to outliers"},
            "Average": {"Formula": "mean pairwise dist", "Shape": "Balanced compromise", "Sensitivity": "Moderate"},
            "Ward": {"Formula": "delta WCSS (variance increase)", "Shape": "Most compact (K-Means-like)", "Sensitivity": "Moderate (industry standard default)"}
        },
        "viva_questions": [
            {"q": "Why doesn't agglomerative clustering require K to be specified in advance?", "a": "Because it builds a complete dendrogram from N singleton leaves to 1 root, encoding every possible partition from K=N to K=1 simultaneously."},
            {"q": "Can a dendrogram ever have a merge height LOWER than one of its children's height?", "a": "For metric linkages (Single, Complete, Average, Ward), merge heights are strictly monotonic (h_parent >= h_child). Centroid or median linkage can cause non-monotonic inversions."},
            {"q": "Why is Ward's linkage often called 'K-Means in disguise'?", "a": "Because both optimize the exact same objective: minimizing Within-Cluster Sum of Squares (WCSS / variance increase)."}
        ]
    }
