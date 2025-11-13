from __future__ import annotations
import numpy as np
import pandas as pd
import os

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist


class CountyClustering:
    def __init__(self, gender: str, immigrants=None, output_dir="output/clusters"):
        self.immigrants = immigrants
        self.output_dir = output_dir
        self.gender = gender
        os.makedirs(self.output_dir, exist_ok=True)

    def _resolve_metric(self, metric: str):
        """Return (scipy_metric_name, alias_note)."""
        metric = (metric or "").strip().lower()
        aliases = {
            "manhattan": "cityblock",
            "l1": "cityblock",
            "euclid": "euclidean",
            "l2": "euclidean",
            "cheby": "chebyshev",
        }
        return aliases.get(metric, metric)

    def _to_numeric_series(self, s: pd.Series) -> pd.Series:
        s = s.astype(str).str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")

    def cluster_on_immigration_data(
        self,
        cluster_threshold: float = 2.0,
        distance_metric: str = "chebyshev",
        linkage_method: str = "complete",
        show_clusters: bool = False,
        mode: str = "percent",
        population: pd.Series = None,
        *,
        # optional kwargs for specific metrics
        minkowski_p: float = 2.0,     # used when metric='minkowski'
        VI: np.ndarray = None,  # inverse covariance for mahalanobis
        cov: np.ndarray = None  # covariance for mahalanobis (if VI not supplied)
    ):
        """
        Cluster counties using hierarchical linkage on immigration values.

        Parameters
        ----------
        cluster_threshold : float
            Distance cut used to form flat clusters.
        distance_metric : str
            Any pdist-compatible metric, e.g. 'euclidean', 'cityblock' (manhattan),
            'canberra', 'chebyshev', 'minkowski', 'cosine', 'correlation',
            'braycurtis', 'mahalanobis', 'hamming', 'jaccard' (boolean).
        linkage_method : str
            Hierarchical linkage method (e.g. 'ward', 'average', 'complete',
            'single', 'weighted', 'median', 'centroid').
            NOTE: 'ward' requires Euclidean distance on observations (not a
            precomputed distance vector).
        mode : {'percent','count'}
            'percent' clusters on % born in Kenya but outside current place;
            'count' clusters on absolute migrant counts (requires population).
        minkowski_p : float
            Order p for Minkowski metric (if used).
        VI, cov : ndarray
            For 'mahalanobis': supply VI (inverse covariance) or covariance matrix.
        """
        # 1) series -> numeric
        immigrants_series = pd.Series(self.immigrants).dropna()
        immigrants_series = self._to_numeric_series(immigrants_series).dropna()

        if mode not in ("percent", "count"):
            raise ValueError("mode must be 'percent' or 'count'")

        # 1b) choose working series
        if mode == "count":
            if population is None:
                raise ValueError("population Series is required when mode='count'")
            pop = self._to_numeric_series(pd.Series(population)).dropna()
            s = immigrants_series.reindex(pop.index)
            # shares may be proportions (<=1.5) or percents (>1.5)
            if s.max(skipna=True) > 1.5:
                s = s / 100.0
            work = (s * pop).dropna()
            value_label = "immigrants count"
        else:
            work = immigrants_series.copy()
            if work.max(skipna=True) <= 1.5:  # proportions → percent
                work = work * 100.0
            value_label = "immigrants percent"

        # 2) DataFrame for pipeline
        df = work.to_frame(name=value_label)

        # 3) Standardize observations
        scaled = StandardScaler().fit_transform(df)

        # 4) Pairwise distances / linkage
        metric = self._resolve_metric(distance_metric)

        # Special handling for Ward: must use euclidean on observations
        if linkage_method.lower() == "ward":
            if metric != "euclidean":
                # silently enforce Euclidean for Ward
                metric = "euclidean"
            Z = linkage(scaled, method="ward", metric="euclidean")
        else:
            # Build kwargs for pdist
            pd_kwargs = {}
            if metric == "minkowski":
                pd_kwargs["p"] = minkowski_p
            if metric == "mahalanobis":
                # If VI not given but covariance is, invert it
                if VI is None and cov is not None:
                    VI = np.linalg.pinv(cov)
                if VI is None:
                    # compute covariance on standardized data if not provided
                    cov = np.cov(scaled.T)
                    VI = np.linalg.pinv(cov)
                pd_kwargs["VI"] = VI

            dists = pdist(scaled, metric=metric, **pd_kwargs)
            # For non-ward methods, passing a condensed distance vector is valid
            Z = linkage(dists, method=linkage_method)

        # 5) Flat clusters
        labels = pd.Series(
            fcluster(Z, t=cluster_threshold, criterion="distance"),
            index=df.index,
            name="cluster"
        )

        # 6) Group print
        clusters = {}
        for county, c_id in labels.items():
            clusters.setdefault(c_id, []).append((county, float(df.loc[county, value_label])))
        for c_id in clusters:
            clusters[c_id].sort(key=lambda x: x[1], reverse=True)

        if show_clusters:
            print(f"\n=== Clusters (threshold={cluster_threshold}) — {self.gender} — {mode} ===")
            print(f"Distance metric: {metric} | Linkage: {linkage_method}")
            for c_id in sorted(clusters):
                members = clusters[c_id]
                print(f"\nCluster {c_id}  (n={len(members)})")
                for county, v in members:
                    suffix = "%" if mode == "percent" else ""
                    print(f"  - {county}: {v:.2f}{suffix}")

        # 7) Save CSV
        out_path = os.path.join(self.output_dir, f"clusters_{self.gender}_{mode}.csv")
        labels.to_frame().join(df).sort_values(
            ["cluster", value_label], ascending=[True, False]
        ).to_csv(out_path, index=True)

        return clusters, labels, Z




