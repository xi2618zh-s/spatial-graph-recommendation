"""Spatial graph construction for Spatial-LightGCN.

Idea (Tobler's first law applied to POI recommendation): nearby POIs share
visitor populations. We add an item-item geographic kNN graph on top of the
user-item interaction graph, which (a) densifies the neighborhood of long-tail
POIs that have few interactions, and (b) injects location awareness without
changing the model at all.

Propagation matrix:

    A_combined = norm(A_bipartite) + lambda * norm(S_geo)

where S_geo is a symmetrized kNN graph over haversine distance with Gaussian
kernel weights, sym-normalized independently, and zero-padded into the
(n_users + n_items) space. `lambda = 0` recovers vanilla LightGCN exactly,
which makes the LightGCN-vs-Spatial comparison a single-knob ablation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0


def load_coords(csv_path: str | Path, n_items: int) -> np.ndarray:
    """Return (n_items, 2) array of [lat, lon] in radians; NaN where unknown."""
    df = pd.read_csv(csv_path)
    coords = np.full((n_items, 2), np.nan, dtype=np.float64)
    ids = df["item_id"].to_numpy()
    coords[ids, 0] = np.radians(df["lat"].to_numpy())
    coords[ids, 1] = np.radians(df["lon"].to_numpy())
    return coords


def geo_knn_graph(coords: np.ndarray, k: int = 10,
                  max_dist_km: float = 100.0) -> tuple[sp.csr_matrix, float]:
    """Symmetric item-item graph: k nearest geographic neighbors per POI,
    edges beyond max_dist_km dropped, values = distance in km.

    Returns (distance graph, median kNN distance in km) — the median is the
    default Gaussian kernel bandwidth.
    """
    n = coords.shape[0]
    valid = ~np.isnan(coords[:, 0])
    idx_valid = np.where(valid)[0]
    tree = BallTree(coords[valid], metric="haversine")
    dist, nbr = tree.query(coords[valid], k=k + 1)  # includes self at col 0
    dist_km = dist[:, 1:] * EARTH_RADIUS_KM
    nbr = idx_valid[nbr[:, 1:]]  # map back to global item ids

    src = np.repeat(idx_valid, k)
    dst = nbr.flatten()
    d = dist_km.flatten()
    keep = d <= max_dist_km
    S = sp.csr_matrix((d[keep], (src[keep], dst[keep])), shape=(n, n))
    S = S.maximum(S.T)  # symmetrize (union of directed kNN edges)
    return S, float(np.median(dist_km))


def gaussian_kernel(S_dist: sp.csr_matrix, sigma_km: float) -> sp.csr_matrix:
    """exp(-(d/sigma)^2) on nonzero entries; closer POIs -> stronger edges."""
    S = S_dist.copy()
    S.data = np.exp(-np.square(S.data / sigma_km)).astype(np.float32)
    return S


def sym_normalize(S: sp.spmatrix) -> sp.csr_matrix:
    deg = np.asarray(S.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5, out=np.zeros_like(deg), where=deg > 0)
    D = sp.diags(d_inv_sqrt)
    return (D @ S @ D).tocsr()


def build_combined_adj(data, coords_csv: str | Path, k: int = 10,
                       lam: float = 0.3, max_dist_km: float = 100.0,
                       sigma_km: float | None = None) -> sp.coo_matrix:
    """norm(bipartite) + lam * norm(geo kNN), in (n_users+n_items)^2 space."""
    A_ui = data.norm_adj().tocsr()

    coords = load_coords(coords_csv, data.n_items)
    S_dist, median_km = geo_knn_graph(coords, k=k, max_dist_km=max_dist_km)
    sigma = sigma_km if sigma_km is not None else median_km
    S = sym_normalize(gaussian_kernel(S_dist, sigma))
    print(f"spatial graph: {S.nnz} edges (k={k}, max_dist={max_dist_km}km, "
          f"sigma={sigma:.2f}km [{'auto-median' if sigma_km is None else 'fixed'}])")

    n_u = data.n_users
    S_pad = sp.bmat(
        [[sp.csr_matrix((n_u, n_u)), None], [None, S]], format="csr"
    )
    return (A_ui + lam * S_pad).tocoo()
