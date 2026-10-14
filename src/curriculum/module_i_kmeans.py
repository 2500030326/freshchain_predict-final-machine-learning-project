"""
Module I: K-Means Clustering & Lloyd's Algorithm
Reference Deck: M4-Session1-KMeans_clustering-new.pptx
Authoritative requirements:
- K-Means objective: Within-Cluster Sum of Squares (WCSS / Inertia) J = sum_k sum_i ||x_i - mu_k||^2
- Lloyd's algorithm: Alternate Assign (nearest centroid) and Update (center of mass)
- K-Means++ distance-weighted probabilistic initialization
- Slide 24 Hand-Calculated Silhouette Example:
    C1 = {A(1,1), B(2,1), C(1,2)}, C2 = {D(5,5), E(6,5)}
    a(A) = 1.000, b(A) = 6.030, s(A) = 0.834
    s(B) = 0.777, s(C) = 0.777, s(D) = 0.808, s(E) = 0.808
    Mean s_bar = 0.801 (Strong cluster structure)
- Slide 18: 9-point trace for K = 1..6 (Elbow curvature and marginal gain)
- Elbow Method vs. Silhouette Score decision workflow
- Terminal Questions Q1-Q8 & Viva Questions Q1-Q10
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist


def solve_slide_24_silhouette_hand_example() -> dict:
    """Exact hand calculation on 5 points from Slide 24."""
    # Points
    A = np.array([1.0, 1.0])
    B = np.array([2.0, 1.0])
    C = np.array([1.0, 2.0])
    D = np.array([5.0, 5.0])
    E = np.array([6.0, 5.0])

    # C1 = {A, B, C}, C2 = {D, E}
    # For A:
    # d(A,B) = 1.0, d(A,C) = 1.0 => a(A) = (1+1)/2 = 1.000
    # d(A,D) = sqrt((5-1)^2 + (5-1)^2) = sqrt(32) approx 5.657
    # d(A,E) = sqrt((6-1)^2 + (5-1)^2) = sqrt(41) approx 6.403
    # b(A) = (5.657 + 6.403)/2 = 6.030
    # s(A) = (6.030 - 1.000)/6.030 = 0.8341
    points = [("A", A, 1), ("B", B, 1), ("C", C, 1), ("D", D, 2), ("E", E, 2)]
    c1 = [A, B, C]
    c2 = [D, E]

    results = []
    for name, pt, cluster in points:
        same_cluster = c1 if cluster == 1 else c2
        other_cluster = c2 if cluster == 1 else c1

        # a(i): mean dist to other points in same cluster
        dists_same = [np.linalg.norm(pt - other) for other in same_cluster if not np.array_equal(pt, other)]
        a_i = float(np.mean(dists_same))

        # b(i): mean dist to points in nearest other cluster
        dists_other = [np.linalg.norm(pt - other) for other in other_cluster]
        b_i = float(np.mean(dists_other))

        # s(i)
        s_i = (b_i - a_i) / max(a_i, b_i)
        results.append({
            "Point": name,
            "Coordinates": f"({int(pt[0])},{int(pt[1])})",
            "Cluster": f"C{cluster}",
            "a_i": round(a_i, 3),
            "b_i": round(b_i, 3),
            "s_i": round(s_i, 3)
        })

    s_mean = float(np.mean([r["s_i"] for r in results]))

    return {
        "points_table": results,
        "mean_silhouette_s_bar": round(s_mean, 3),
        "interpretation": "s_bar ≈ 0.801 -> Strong cluster structure (within 0.71 – 1.00 range)."
    }


def compute_elbow_and_silhouette(X: np.ndarray, k_range=range(2, 7)) -> list[dict]:
    """Computes WCSS and Silhouette score across candidate K."""
    res = []
    wcss_list = []
    for k in range(1, max(k_range) + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(X)
        wcss_list.append(km.inertia_)

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(X)
        sil = float(silhouette_score(X, km.labels_))
        wcss = wcss_list[k - 1]
        prev_wcss = wcss_list[k - 2]
        marginal_gain = prev_wcss - wcss

        res.append({
            "K": k,
            "WCSS_Inertia": round(float(wcss), 2),
            "Marginal_Gain_Reduction": round(float(marginal_gain), 2),
            "Silhouette_Score": round(sil, 4)
        })
    return res


def run_module_i_walkthrough() -> dict:
    # 1. Slide 24 hand calculation
    hand_calc = solve_slide_24_silhouette_hand_example()

    # 2. Slide 18 9-point setup (3 natural groups)
    X_9pts = np.array([
        [1.0, 1.0], [1.5, 2.0], [2.0, 1.0],      # Group 1
        [8.0, 8.0], [8.5, 9.0], [9.0, 8.5],      # Group 2
        [1.0, 9.0], [1.5, 9.5], [2.0, 9.0]       # Group 3
    ])
    eval_9pts = compute_elbow_and_silhouette(X_9pts, k_range=range(2, 6))

    return {
        "slide24_hand_calculation": hand_calc,
        "elbow_and_silhouette_trace": eval_9pts,
        "comparison_techniques": {
            "Lloyd's Algorithm": {"Purpose": "Optimise assignments for fixed K", "Metric": "WCSS J", "Cost": "O(nKd)/iter"},
            "Elbow Method": {"Purpose": "Select optimal K via bend", "Metric": "WCSS vs K curve", "Cost": "Run K_max times"},
            "Silhouette Score": {"Purpose": "Select K + evaluate quality", "Metric": "s(i) in [-1, +1]", "Cost": "O(n^2)"}
        },
        "viva_questions": [
            {"q": "What does the ‘K’ in K-Means represent, and who chooses it?", "a": "K represents the predefined number of clusters, chosen by the practitioner before running the algorithm."},
            {"q": "What happens to WCSS as K approaches n (number of data points)?", "a": "WCSS drops monotonically to 0, because every single point becomes its own centroid."},
            {"q": "Why does random initialisation sometimes give a worse clustering than k-means++?", "a": "Random initialisation can pick centroids in the same cluster, leading to poor local minima. K-means++ spreads out initial seeds using distance-weighted probability."},
            {"q": "If a(i) > b(i) for a point, what does that tell you about its cluster assignment?", "a": "s(i) becomes negative, meaning the point is closer to a neighboring cluster than its assigned cluster (likely misclassified)."}
        ]
    }
