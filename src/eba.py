from __future__ import annotations
import itertools
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class EBASpec:
    """Holds one model specification and its results for a single focus variable."""
    spec_id: int
    free_vars: Tuple[str, ...]
    focus_var: str
    doubtful_vars: Tuple[str, ...]
    coef: float
    se: float
    pval: float
    adj_r2: float


class ExtremeBoundsAnalyzer:
    """
    Extreme Bounds Analysis for OLS models following:
      - Leamer (1985) 'extreme bounds' (sign stability),
      - Sala-i-Martin (1997) CDF(0) (distribution-based robustness).

    Apply this AFTER backward elimination to the set of *selected* variables
    (plus any theory-mandated 'free' variables like log distance, log population).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        y_col: str,
        free_vars: Iterable[str],
        focus_vars: Iterable[str],
        doubtful_pool: Iterable[str],
        output_dir: str = "output/eba",
        max_k: int = 3,
        robust: bool = True,
        weight_by: str = "adj_r2",  # {"none", "adj_r2"}
        random_subsample: Optional[int] = None,  # if huge combinations, sample N specs
        alpha: float = 0.05,
    ):
        self.df = df.copy()
        self.y_col = y_col
        self.free_vars = tuple(dict.fromkeys(free_vars))          # dedupe keep order
        self.focus_vars = tuple(dict.fromkeys(focus_vars))
        # Remove focus+free from pool
        pool = [v for v in doubtful_pool if v not in self.free_vars and
                v not in self.focus_vars]
        self.doubtful_pool = tuple(dict.fromkeys(pool))
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_k = int(max_k)
        self.robust = robust
        self.weight_by = weight_by
        self.random_subsample = random_subsample
        self.alpha = alpha

        # basic sanity
        cols_needed = set([self.y_col]) | set(self.free_vars) | \
                      set(self.focus_vars) | set(self.doubtful_pool)
        missing = [c for c in cols_needed if c not in self.df.columns]
        if missing:
            raise ValueError(f"EBA missing columns in DataFrame: {missing}")

        # Drop rows with any NA used in any spec (simplest + consistent sample)
        self.df = self.df[list(cols_needed)].dropna().copy()

    # ---------- helpers ----------

    def _fit_ols(self, X_cols: List[str]) -> sm.regression.linear_model.RegressionResultsWrapper:
        X = sm.add_constant(self.df[X_cols], has_constant='add')
        y = self.df[self.y_col].values
        if self.robust:
            # HC1 (Eicker-White, Stata default) is a sensible choice
            return sm.OLS(y, X).fit(cov_type="HC1")
        return sm.OLS(y, X).fit()

    def _iterate_specs_for_focus(self, focus: str) -> List[EBASpec]:
        specs: List[EBASpec] = []
        spec_id = 0

        all_doubtful = list(self.doubtful_pool)
        # Build combination iterator up to size max_k
        combinations = []
        for k in range(0, min(self.max_k, len(all_doubtful)) + 1):
            combinations.extend(list(itertools.combinations(all_doubtful, k)))

        # Optional subsample to curb combinatorial explosion
        if self.random_subsample is not None and self.random_subsample < len(combinations):
            rng = np.random.default_rng(42)
            idx = rng.choice(len(combinations), size=self.random_subsample, replace=False)
            combinations = [combinations[i] for i in idx]

        for comb in combinations:
            X_cols = list(self.free_vars) + [focus] + list(comb)
            res = self._fit_ols(X_cols)

            # grab the focus coefficient
            if focus not in res.params.index:
                # Shouldn't happen if columns are there, but guard anyway
                continue

            specs.append(
                EBASpec(
                    spec_id=spec_id,
                    free_vars=tuple(self.free_vars),
                    focus_var=focus,
                    doubtful_vars=tuple(comb),
                    coef=float(res.params[focus]),
                    se=float(res.bse[focus]),
                    pval=float(res.pvalues[focus]),
                    adj_r2=float(res.rsquared_adj),
                )
            )
            spec_id += 1

        return specs

    # ---------- Leamer EBA ----------

    @staticmethod
    def _extreme_bounds(specs: List[EBASpec], t_crit: float = 1.96) -> Dict[str, float]:
        """
        For Leamer's EBA, compute lower and upper 'extreme bounds' across specs:
          lower = min (coef - t * se)
          upper = max (coef + t * se)
        """
        lows = [s.coef - t_crit * s.se for s in specs]
        ups  = [s.coef + t_crit * s.se for s in specs]
        return {"lower": float(np.min(lows)), "upper": float(np.max(ups))}

    @staticmethod
    def _leamer_robust(bounds: Dict[str, float]) -> bool:
        """Robust if bounds do not straddle zero (strict sign stability)."""
        return bounds["lower"] > 0 or bounds["upper"] < 0

    # ---------- Sala-i-Martin EBA ----------

    @staticmethod
    def _cdf_zero_for_spec(coef: float, se: float) -> float:
        """Prob(coef > 0) under Normal approx; return mass on sign of mean."""
        if se <= 0 or not np.isfinite(se):
            return 1.0 if coef != 0 else 0.5
        z = coef / se
        # Phi(z)
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        # mass on the side of the estimate's sign
        return float(cdf if coef >= 0 else (1.0 - cdf))

    @staticmethod
    def _weights(specs: List[EBASpec], mode: str) -> np.ndarray:
        if mode == "adj_r2":
            w = np.array([max(0.0, s.adj_r2) for s in specs], dtype=float)
            if w.sum() == 0:
                return np.ones(len(specs)) / len(specs)
            return w / w.sum()
        # 'none'
        return np.ones(len(specs)) / len(specs)

    def _sala_i_martin(self, specs: List[EBASpec]) -> Dict[str, float]:
        """
        Weighted average of coefficients and their SE's; compute CDF(0) following
        Sala-i-Martin’s idea (probability mass on one side of zero).
        """
        w = self._weights(specs, self.weight_by)
        coefs = np.array([s.coef for s in specs], dtype=float)
        ses   = np.array([s.se   for s in specs], dtype=float)

        # Weighted mean of betas
        beta_bar = float(np.sum(w * coefs))
        # Two common options exist for the variance; a simple, conservative proxy:
        # average of squared SEs + between-spec variance (Delta method flavor)
        within = float(np.sum(w * (ses ** 2)))
        between = float(np.sum(w * ((coefs - beta_bar) ** 2)))
        var_beta = within + between
        se_beta = float(np.sqrt(max(var_beta, 1e-12)))

        # Mass on the side of the mean (Normal approx)
        cdf0 = self._cdf_zero_for_spec(beta_bar, se_beta)
        return {
            "beta_bar": beta_bar,
            "se_beta": se_beta,
            "cdf0": cdf0,
        }

    # ---------- Public API ----------

    def analyze(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
          - leamer_df: one-row per focus var with extreme bounds & robust flag
          - sim_df:    one-row per focus var with beta_bar, se_beta, CDF(0)
        Also writes per-focus CSV of all specs.
        """
        leamer_rows = []
        sim_rows = []

        for focus in self.focus_vars:
            specs = self._iterate_specs_for_focus(focus)

            if len(specs) == 0:
                continue

            # Save all specs for this focus variable
            all_specs_df = pd.DataFrame([s.__dict__ for s in specs])
            all_specs_path = os.path.join(self.output_dir, f"eba_specs_{focus}.csv")
            all_specs_df.to_csv(all_specs_path, index=False)

            # Leamer
            bounds = self._extreme_bounds(specs)
            robust = self._leamer_robust(bounds)
            leamer_rows.append({
                "variable": focus,
                "lower_95": bounds["lower"],
                "upper_95": bounds["upper"],
                "leamer_robust": bool(robust),
                "n_specs": len(specs),
            })

            # Sala-i-Martin
            sim = self._sala_i_martin(specs)
            sim_rows.append({
                "variable": focus,
                "beta_bar": sim["beta_bar"],
                "se_beta": sim["se_beta"],
                "CDF0": sim["cdf0"],
                "weighted_by": self.weight_by,
                "n_specs": len(specs),
            })

        leamer_df = pd.DataFrame(leamer_rows).sort_values("variable")
        sim_df = pd.DataFrame(sim_rows).sort_values("variable")

        # Save summary CSVs
        leamer_df.to_csv(os.path.join(self.output_dir, "eba_leamer_summary.csv"),
                         index=False)
        sim_df.to_csv(os.path.join(self.output_dir, "eba_sim_summary.csv"),
                      index=False)

        return leamer_df, sim_df
