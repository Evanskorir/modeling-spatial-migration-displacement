"""First screening stage: predictor correlations, separately by sex."""
from pathlib import Path
import numpy as np
import pandas as pd
from .plotter import Plotter


class CorrelationAnalysis:
    """Iteratively resolve predictor pairs with |Pearson r| above threshold.

    Outcome correlations never determine selection. For the strongest flagged
    pair retain the earlier variable in the common, documented priority list.
    """
    def __init__(self, output_dir, threshold=.80):
        if not 0 < threshold < 1:
            raise ValueError("Correlation threshold must be between 0 and 1")
        self.output_dir, self.threshold = Path(output_dir), threshold

    def run(self, data, features, sex, source_data=None):
        folder = self.output_dir / sex
        folder.mkdir(parents=True, exist_ok=True)
        features = list(features)
        current = features.copy()
        x = data[features]
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError("Nonfinite correlation input")
        if len(features) != len(set(features)) or not features:
            raise ValueError('Predictor names must be nonempty and unique')
        if (x.std(ddof=0) <= 1e-12).any():
            raise ValueError('Constant correlation predictor')
        x.corr().to_csv(folder / "initial_correlation_matrix.csv")
        if source_data is not None:
            columns = [c for c in source_data if c not in ["County", "county_key", "sex", "y"]]
            source_data[columns].corr().to_csv(folder / "all_source_predictor_correlations.csv")
        pd.DataFrame([dict(variable=c, pearson_r_with_outcome=x[c].corr(data.y),
                           used_for_selection=False) for c in features]).to_csv(
            folder / "outcome_correlations_descriptive_only.csv", index=False)
        pairs, history = [], []
        initial_corr = x.corr()
        for i, first in enumerate(features):
            for second in features[i + 1:]:
                r = float(initial_corr.loc[first, second])
                pairs.append(dict(first=first, second=second, correlation=r,
                                  exceeds_threshold=abs(r) > self.threshold))
        while len(current) > 1:
            corr = data[current].corr().abs()
            candidates = [(float(corr.loc[a, b]), a, b) for i, a in enumerate(current)
                          for b in current[i + 1:]]
            maximum, first, second = max(candidates, key=lambda item: item[0])
            if maximum <= self.threshold + 1e-12:
                break
            score = (corr.sum(axis=1) - 1) / (len(current) - 1)
            dropped = max([first, second], key=features.index)
            kept = second if dropped == first else first
            signed = float(data[first].corr(data[second]))
            history.append(dict(iteration=len(history) + 1, pair_first=first, pair_second=second,
                                correlation=signed, dropped=dropped, paired_variable_kept=kept,
                                first_mean_abs_r=score[first], second_mean_abs_r=score[second],
                                threshold=self.threshold))
            current.remove(dropped)
        pd.DataFrame(pairs).to_csv(folder / "correlated_pairs.csv", index=False)
        pd.DataFrame(history, columns=["iteration", "pair_first", "pair_second", "correlation",
                                      "dropped", "paired_variable_kept", "first_mean_abs_r",
                                      "second_mean_abs_r", "threshold"]).to_csv(folder / "removal_history.csv", index=False)
        pd.DataFrame(dict(variable=features, retained_after_correlation=[c in current for c in features])).to_csv(
            folder / "variable_decisions.csv", index=False)
        data[current].corr().to_csv(folder / "retained_correlation_matrix.csv")
        plotter = Plotter(folder / 'figures')
        plotter.correlation(initial_corr, 'initial_correlation_heatmap')
        plotter.correlation(data[current].corr(), 'retained_correlation_heatmap')
        return current

