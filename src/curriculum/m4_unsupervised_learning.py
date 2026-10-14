"""
Module 4: Unsupervised Learning
================================
Complete, self-contained runnable implementations covering:
1. K-Means & Lloyd's algorithm from scratch (Assignment & Update loop).
2. Elbow method (Inertia curve) and Silhouette score calculation.
3. K-Means++ initialization algorithm & Mini-Batch K-Means comparison.
4. Agglomerative Hierarchical Clustering: Linkages (Single, Complete, Average, Ward) & Cophenetic correlation.
5. DBSCAN from scratch/sklearn: eps, minPts, Core/Border/Noise identification on non-convex manifolds.
6. PCA in production: Standardization impact, explained variance ratio, and unstandardized pitfalls.
7. t-SNE / Manifold visualization principles (why distances are non-metric).
8. Anomaly detection: Isolation Forest, One-Class SVM, and Reconstruction Residuals.
9. Practical pipeline: Cluster labels & distances as engineered features for a supervised model.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons, make_classification
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist


# =====================================================================
# 1. K-MEANS & LLOYD'S ALGORITHM FROM SCRATCH
# =====================================================================

class LloydKMeansScratch:
    """Explicit implementation of Lloyd's 1982 algorithm."""

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X: np.ndarray):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # Step 0: Randomly initialize centroids from points
        init_idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[init_idx].copy()

        for it in range(self.max_iter):
            # Step 1 (Assignment): Compute pairwise squared Euclidean distances: ||x_i - mu_k||^2
            # Shape: (n_samples, n_clusters)
            dists = np.sum((X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)

            # Step 2 (Update): Recompute centroids as cluster mean
            new_centers = np.zeros_like(self.cluster_centers_)
            for k in range(self.n_clusters):
                assigned = X[labels == k]
                if len(assigned) > 0:
                    new_centers[k] = assigned.mean(axis=0)
                else:
                    new_centers[k] = self.cluster_centers_[k]

            # Check convergence
            center_shift = np.max(np.linalg.norm(new_centers - self.cluster_centers_, axis=1))
            self.cluster_centers_ = new_centers
            if center_shift < self.tol:
                break

        # Compute final inertia (Within-Cluster Sum of Squares)
        min_dists = np.min(np.sum((X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]) ** 2, axis=2), axis=1)
        self.inertia_ = float(np.sum(min_dists))
        return self


def demo_kmeans_and_model_selection():
    print("\n" + "=" * 60)
    print("DEMO 1 & 2: K-MEANS, ELBOW INERTIA, AND SILHOUETTE SCORE")
    print("=" * 60)

    X, _ = make_blobs(n_samples=600, centers=4, n_features=4, cluster_std=1.2, random_state=42)

    # Scratch verification
    scratch_km = LloydKMeansScratch(n_clusters=4, random_state=42).fit(X)
    print(f"Scratch Lloyd's K-Means Inertia (K=4): {scratch_km.inertia_:.2f}")

    # Evaluating K via Elbow (Inertia) and Silhouette Scores
    results = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(X)
        sil = silhouette_score(X, km.labels_)
        results.append({"K": k, "Inertia": round(km.inertia_, 2), "Silhouette_Score": round(sil, 4)})

    eval_df = pd.DataFrame(results)
    print("\n--- Model Selection Table: Elbow & Silhouette ---")
    print(eval_df.to_string(index=False))
    best_k = eval_df.loc[eval_df["Silhouette_Score"].idxmax(), "K"]
    print(f"\n[Optimal K by Silhouette]: K = {int(best_k)}")


# =====================================================================
# 2. K-MEANS++ & MINI-BATCH K-MEANS COMPARISON
# =====================================================================

def demo_kmeans_plus_plus_and_minibatch():
    print("\n" + "=" * 60)
    print("DEMO 3: K-MEANS++ AND MINI-BATCH K-MEANS ON LARGE DATA")
    print("=" * 60)

    import time

    # Generate 50,000 samples
    X_large, _ = make_blobs(n_samples=50000, centers=5, n_features=10, random_state=42)

    # Standard K-Means (Lloyd)
    t0 = time.time()
    km_standard = KMeans(n_clusters=5, init="k-means++", n_init=1, max_iter=100, random_state=42).fit(X_large)
    t_standard = time.time() - t0

    # Mini-Batch K-Means
    t0 = time.time()
    km_minibatch = MiniBatchKMeans(n_clusters=5, batch_size=2048, max_iter=100, random_state=42).fit(X_large)
    t_minibatch = time.time() - t0

    print(f"Standard K-Means:   Runtime = {t_standard:.4f}s | Inertia = {km_standard.inertia_:.2f}")
    print(f"Mini-Batch K-Means: Runtime = {t_minibatch:.4f}s | Inertia = {km_minibatch.inertia_:.2f}")
    print(f"Speedup Factor:     {t_standard / max(1e-5, t_minibatch):.2f}x")


# =====================================================================
# 3. AGGLOMERATIVE HIERARCHICAL CLUSTERING & LINKAGE
# =====================================================================

def demo_hierarchical_clustering():
    print("\n" + "=" * 60)
    print("DEMO 4: HIERARCHICAL CLUSTERING & LINKAGE COMPARISON")
    print("=" * 60)

    X_small, _ = make_blobs(n_samples=50, centers=3, n_features=2, random_state=42)
    condensed_dist = pdist(X_small)

    linkage_methods = ["single", "complete", "average", "ward"]
    for method in linkage_methods:
        Z = linkage(X_small, method=method)
        c, _ = cophenet(Z, condensed_dist)
        print(f"Linkage Method: {method:<10} | Cophenetic Correlation Coefficient: {c:.4f}")
    print("Note: Cophenetic correlation measures how faithfully the dendrogram preserves pairwise distances.")


# =====================================================================
# 4. DBSCAN & NON-CONVEX CLUSTER DISCOVERY
# =====================================================================

def demo_dbscan_non_convex():
    print("\n" + "=" * 60)
    print("DEMO 5: DBSCAN (DENSITY-BASED SPATIAL CLUSTERING)")
    print("=" * 60)

    # Concentric non-linear moons with noise
    X_moons, _ = make_moons(n_samples=400, noise=0.08, random_state=42)

    # 1. K-Means fails on non-convex manifolds
    km = KMeans(n_clusters=2, random_state=42).fit(X_moons)

    # 2. DBSCAN succeeds without requiring predefined K
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    labels = dbscan.fit_predict(X_moons)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f"DBSCAN Parameters: eps=0.2, minPts=5")
    print(f"Clusters Discovered: {n_clusters_}")
    print(f"Noise / Anomaly Points Detected: {n_noise_} ({n_noise_ / len(X_moons) * 100:.1f}%)")
    print("DBSCAN successfully clusters arbitrary non-convex geometries and isolates outliers naturally.")


# =====================================================================
# 5. PCA IN PRODUCTION: THE CRITICAL STANDARDIZATION REQUIREMENT
# =====================================================================

def demo_pca_standardization_pitfall():
    print("\n" + "=" * 60)
    print("DEMO 6: PCA PRODUCTION USAGE & STANDARDIZATION PITFALL")
    print("=" * 60)

    np.random.seed(42)
    # Feature 1 has low variance (scale ~1), Feature 2 has artificially massive scale (~10,000)
    feature_1 = np.random.normal(0, 1.0, 500)
    feature_2 = np.random.normal(0, 10000.0, 500)
    feature_3 = np.random.normal(0, 2.0, 500)
    X_unscaled = np.column_stack([feature_1, feature_2, feature_3])

    # Case A: PCA without standardization
    pca_unscaled = PCA(n_components=3)
    pca_unscaled.fit(X_unscaled)

    # Case B: PCA with standardization (Production Standard)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unscaled)
    pca_scaled = PCA(n_components=3)
    pca_scaled.fit(X_scaled)

    print("--- Unstandardized PCA Explained Variance Ratio ---")
    print(f"PC1: {pca_unscaled.explained_variance_ratio_[0]:.6f} (Artificially captures ~100% variance purely due to units!)")
    print(f"PC2: {pca_unscaled.explained_variance_ratio_[1]:.6f}")

    print("\n--- Standardized PCA Explained Variance Ratio (Correct) ---")
    print(f"PC1: {pca_scaled.explained_variance_ratio_[0]:.4f}")
    print(f"PC2: {pca_scaled.explained_variance_ratio_[1]:.4f}")
    print(f"PC3: {pca_scaled.explained_variance_ratio_[2]:.4f}")


# =====================================================================
# 6. MANIFOLD VISUALIZATION: t-SNE HYPERPARAMETERS & NON-METRIC DISTANCES
# =====================================================================

def demo_tsne_manifold():
    print("\n" + "=" * 60)
    print("DEMO 7: t-SNE VISUALIZATION WORKHORSE")
    print("=" * 60)

    X_digits, _ = make_blobs(n_samples=300, centers=3, n_features=12, random_state=42)

    # t-SNE projection
    tsne = TSNE(n_components=2, perplexity=30.0, max_iter=500, random_state=42)
    embedding = tsne.fit_transform(X_digits)

    print(f"t-SNE Output Embedding Shape: {embedding.shape}")
    print("Key Axiom: Inter-cluster distances in t-SNE/UMAP space DO NOT reflect metric distances!")
    print("Local neighborhoods are preserved; global spacing is arbitrary due to the non-linear objective.")


# =====================================================================
# 7. ANOMALY DETECTION: ISOLATION FOREST, ONE-CLASS SVM & RESIDUALS
# =====================================================================

def demo_anomaly_detection():
    print("\n" + "=" * 60)
    print("DEMO 8: ANOMALY DETECTION PARADIGMS")
    print("=" * 60)

    np.random.seed(42)
    # Generate 450 inliers and 50 outliers
    X_inliers = np.random.normal(0, 1.0, size=(450, 4))
    X_outliers = np.random.uniform(low=-8, high=8, size=(50, 4))
    X_all = np.vstack([X_inliers, X_outliers])
    ground_truth = np.array([1] * 450 + [-1] * 50)  # 1: inlier, -1: outlier

    # 1. Isolation Forest
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso_preds = iso.fit_predict(X_all)
    iso_detected = np.sum((iso_preds == -1) & (ground_truth == -1))

    # 2. One-Class SVM
    oc_svm = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
    svm_preds = oc_svm.fit_predict(X_all)
    svm_detected = np.sum((svm_preds == -1) & (ground_truth == -1))

    # 3. PCA Reconstruction Residuals (Linear Autoencoder Intuition)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_reduced)
    recon_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
    threshold = np.percentile(recon_error, 90)
    recon_preds = np.where(recon_error > threshold, -1, 1)
    recon_detected = np.sum((recon_preds == -1) & (ground_truth == -1))

    print(f"Total True Outliers: 50")
    print(f"Isolation Forest Detected:    {iso_detected}/50 ({iso_detected/50*100:.1f}%)")
    print(f"One-Class SVM Detected:       {svm_detected}/50 ({svm_detected/50*100:.1f}%)")
    print(f"Reconstruction Error Detected:{recon_detected}/50 ({recon_detected/50*100:.1f}%)")


# =====================================================================
# 8. PRACTICAL PIPELINE: CLUSTER LABELS AS DOWNSTREAM FEATURES
# =====================================================================

def demo_clustering_as_feature_engineering():
    print("\n" + "=" * 60)
    print("DEMO 9: CLUSTER FEATURES IN DOWNSTREAM SUPERVISED MODEL")
    print("=" * 60)

    # Complex non-linear classification dataset
    X, y = make_classification(
        n_samples=1500, n_features=6, n_informative=4, n_redundant=1, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Baseline: Linear Logistic Regression on raw features
    lr_base = LogisticRegression(random_state=42)
    lr_base.fit(X_train, y_train)
    base_acc = lr_base.score(X_test, y_test)

    # Enhanced: Fit K-Means on Train set, compute cluster distances as new features
    kmeans = KMeans(n_clusters=6, random_state=42).fit(X_train)
    train_dist_features = kmeans.transform(X_train)
    test_dist_features = kmeans.transform(X_test)

    X_train_augmented = np.hstack([X_train, train_dist_features])
    X_test_augmented = np.hstack([X_test, test_dist_features])

    lr_augmented = LogisticRegression(random_state=42, max_iter=500)
    lr_augmented.fit(X_train_augmented, y_train)
    aug_acc = lr_augmented.score(X_test_augmented, y_test)

    print(f"Baseline Logistic Regression Test Accuracy:          {base_acc:.4f}")
    print(f"Logistic Regression + Cluster Distance Features Acc: {aug_acc:.4f}")
    print(f"Performance Delta:                                   {aug_acc - base_acc:+.4f}")


if __name__ == "__main__":
    demo_kmeans_and_model_selection()
    demo_kmeans_plus_plus_and_minibatch()
    demo_hierarchical_clustering()
    demo_dbscan_non_convex()
    demo_pca_standardization_pitfall()
    demo_tsne_manifold()
    demo_anomaly_detection()
    demo_clustering_as_feature_engineering()
    print("\n[SUCCESS] Module 4 execution finished successfully.")
