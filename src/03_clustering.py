import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.cluster.hierarchy import linkage, dendrogram


# ============================================================
# PREPARACIÓN DE DATOS
# ============================================================

def preparar_datos_clustering(df_liga: pd.DataFrame):
    """
    Selecciona partidos 1–14 y variables de probabilidad (1X2).
    """
    df_cluster = df_liga[df_liga["num"].between(1, 14)][
        ["probabilidad1", "probabilidadX", "probabilidad2"]
    ].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    return df_cluster, X_scaled, scaler


# ============================================================
# CLUSTERING JERÁRQUICO (EXPLORATORIO)
# ============================================================

def plot_dendrograma(X_scaled):
    """
    Dendrograma usando método Ward.
    Útil para explorar estructura de clusters.
    """
    Z = linkage(X_scaled, method="ward")

    plt.figure(figsize=(12, 6))
    dendrogram(Z, leaf_rotation=45, leaf_font_size=10)

    plt.axhline(y=10, color="red", linestyle="--", alpha=0.7)
    plt.title("Dendrograma – Clustering jerárquico (Ward)")
    plt.xlabel("Clusters")
    plt.ylabel("Distancia")
    plt.tight_layout()
    plt.show()


# ============================================================
# PCA
# ============================================================

def aplicar_pca(X_scaled):
    """
    Reduce dimensionalidad y permite interpretar estructura.
    """
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pd.DataFrame({
        "Componente": ["PC1", "PC2", "PC3"],
        "Varianza explicada": pca.explained_variance_ratio_,
        "Varianza acumulada": pca.explained_variance_ratio_.cumsum()
    })

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2", "PC3"],
        index=["probabilidad1", "probabilidadX", "probabilidad2"]
    )

    return X_pca, explained_var, loadings


def plot_pca(X_pca):
    """
    Visualización en 2D.
    """
    df_pca = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])

    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca["PC1"], df_pca["PC2"], alpha=0.5)

    plt.axhline(0, color="grey", lw=0.5)
    plt.axvline(0, color="grey", lw=0.5)

    plt.xlabel("PC1 – Claridad del favorito")
    plt.ylabel("PC2 – Peso del empate")
    plt.title("PCA de partidos según probabilidades 1–X–2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df_pca


# ============================================================
# SELECCIÓN DE k
# ============================================================

def evaluar_kmeans(X_scaled):
    """
    Método del codo + silhouette.
    """
    wcss = []
    silhouette_avgs = []

    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=25)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

        if k > 1:
            labels = kmeans.labels_
            silhouette_avgs.append(silhouette_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, 11), wcss, marker="o")
    axes[0].set_title("Método del codo")

    axes[1].plot(range(2, 11), silhouette_avgs, marker="o")
    axes[1].set_title("Silhouette score")

    plt.tight_layout()
    plt.show()


# ============================================================
# KMEANS
# ============================================================

def aplicar_kmeans(df_cluster, X_scaled, k):
    """
    Aplica KMeans y devuelve dataframe con clusters.
    """
    model = KMeans(n_clusters=k, random_state=42, n_init=25)
    labels = model.fit_predict(X_scaled)

    df_out = df_cluster.copy()
    df_out["cluster"] = labels

    return df_out, labels, model


def plot_clusters(df_cluster, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df_cluster["probabilidad1"],
        df_cluster["probabilidad2"],
        c=labels,
        cmap="Set1",
        alpha=0.6
    )

    plt.xlabel("Probabilidad 1 (%)")
    plt.ylabel("Probabilidad 2 (%)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# GMM
# ============================================================

def aplicar_gmm(X_scaled, k):
    """
    Gaussian Mixture Model.
    """
    gmm = GaussianMixture(n_components=k, random_state=42)
    labels = gmm.fit_predict(X_scaled)
    proba = gmm.predict_proba(X_scaled)

    return labels, proba


# ============================================================
# DBSCAN
# ============================================================

def aplicar_dbscan(X_scaled, eps=0.35, min_samples=5):
    """
    DBSCAN para detección de outliers.
    eps debe ajustarse según k-distance plot.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    return labels


def plot_k_distance(X_scaled, min_samples=5):
    """
    Ayuda a elegir eps.
    """
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_scaled)

    distances, _ = nn.kneighbors(X_scaled)
    kdist = np.sort(distances[:, -1])

    plt.figure(figsize=(8, 4))
    plt.plot(kdist)
    plt.title("k-distance plot")
    plt.grid(alpha=0.3)
    plt.show()


# ============================================================
# INTERPRETACIÓN DE CLUSTERS
# ============================================================

def analizar_clusters(df_liga, df_cluster, labels):
    """
    Une clusters con dataset original y analiza distribución.
    """
    df_cluster_out = df_cluster.copy()
    df_cluster_out["cluster"] = labels

    df_liga_cluster = df_liga.merge(
        df_cluster_out[["cluster"]],
        left_index=True,
        right_index=True,
        how="inner"
    )

    print("\n===== DISTRIBUCIÓN DE SIGNOS POR CLUSTER =====")
    print(
        df_liga_cluster
        .groupby(["cluster", "signo"])
        .size()
        .unstack(fill_value=0)
    )


# ============================================================
# PIPELINE
# ============================================================

def main(df_liga):
    df_cluster, X_scaled, scaler = preparar_datos_clustering(df_liga)

    plot_dendrograma(X_scaled)

    X_pca, explained_var, loadings = aplicar_pca(X_scaled)
    df_pca = plot_pca(X_pca)

    evaluar_kmeans(X_scaled)

    # KMeans k=3
    df_k3, labels_3, model_3 = aplicar_kmeans(df_cluster, X_scaled, k=3)
    plot_clusters(df_k3, labels_3, "KMeans k=3")

    # KMeans k=4
    df_k4, labels_4, model_4 = aplicar_kmeans(df_cluster, X_scaled, k=4)
    plot_clusters(df_k4, labels_4, "KMeans k=4")

    # GMM
    gmm_labels, _ = aplicar_gmm(X_scaled, k=3)

    # DBSCAN
    plot_k_distance(X_scaled)
    db_labels = aplicar_dbscan(X_scaled)

    analizar_clusters(df_liga, df_cluster, labels_3)


if __name__ == "__main__":
    print("Este módulo debe ejecutarse desde un pipeline con df_liga.")
