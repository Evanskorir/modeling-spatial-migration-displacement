"""Second screening stage: iterative VIF after correlation screening."""
from pathlib import Path
import numpy as np
import pandas as pd
from .plotter import Plotter


def calculate_vif(data, features):
    rows = []
    for key in features:
        y = data[key].to_numpy(dtype=float)
        others = [c for c in features if c != key]
        x = np.column_stack([np.ones(len(data)), data[others].to_numpy(dtype=float)])
        residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        sst = np.sum((y - y.mean()) ** 2)
        if sst <= 1e-20:
            value = np.inf
        else:
            ratio = np.sum(residual ** 2) / sst
            value = np.inf if ratio < 1e-12 else max(1., 1. / ratio)
        rows.append(dict(variable=key, vif=value))
    return pd.DataFrame(rows, columns=["variable", "vif"])


class VIFAnalysis:
    def __init__(self, output_dir, threshold=5.):
        if threshold <= 1:
            raise ValueError("VIF threshold must exceed 1")
        self.output_dir, self.threshold = Path(output_dir), float(threshold)

    def run(self, data, features, sex, initial_features=None, make_plots=True):
        folder = self.output_dir / sex
        folder.mkdir(parents=True, exist_ok=True)
        initial_features=list(features) if initial_features is None else list(initial_features)
        if len(initial_features)!=len(set(initial_features)) or not set(features).issubset(initial_features):
            raise ValueError('VIF initial features must be unique and contain the screened set')
        if not np.isfinite(data[initial_features].to_numpy(float)).all():
            raise ValueError('Nonfinite VIF input')
        initial_values=calculate_vif(data,initial_features)
        initial_values.to_csv(folder/'initial_candidate_vif.csv',index=False)
        current, history, removals = list(features), [], []
        iteration = 0
        while current:
            values = calculate_vif(data, current)
            values["iteration"] = iteration
            values["threshold"] = self.threshold
            history.append(values.copy())
            maximum = values.vif.max()
            if maximum <= self.threshold + 1e-10:
                break
            # List order breaks ties reproducibly; the outcome is never consulted.
            dropped = values.loc[values.vif.idxmax(), "variable"]
            removals.append(dict(iteration=iteration, dropped=dropped, vif=maximum,
                                 remaining_before_removal=len(current)))
            current.remove(dropped)
            iteration += 1
        if not current:
            raise ValueError("VIF screening removed every variable; inspect inputs")
        pd.concat(history, ignore_index=True).to_csv(folder / "vif_iteration_history.csv", index=False)
        history[0].to_csv(folder / "initial_vif_after_correlation.csv", index=False)
        calculate_vif(data, current).to_csv(folder / "final_vif.csv", index=False)
        pd.DataFrame(removals, columns=["iteration", "dropped", "vif", "remaining_before_removal"]).to_csv(
            folder / "removal_history.csv", index=False)
        pd.DataFrame(dict(variable=list(features), retained_after_vif=[c in current for c in features])).to_csv(
            folder / "variable_decisions.csv", index=False)
        pd.DataFrame(dict(variable=current)).to_csv(folder / "retained_variables.csv", index=False)
        x = np.column_stack([np.ones(len(data)), data[current].to_numpy()])
        if np.linalg.matrix_rank(x) != x.shape[1]:
            raise ValueError("Final VIF-screened design is rank deficient")
        initial_lookup=initial_values.set_index('variable').vif.to_dict()
        final_lookup=calculate_vif(data,current).set_index('variable').vif.to_dict()
        removed_lookup={r['dropped']:r['vif'] for r in removals}
        diagnostics=[]
        for variable in initial_features:
            status=('retained' if variable in current else
                    'omitted_correlation' if variable not in features else 'omitted_vif')
            diagnostics.append(dict(sex=sex,variable=variable,initial_vif=initial_lookup[variable],
                                    final_vif=final_lookup.get(variable,np.nan),status=status,
                                    removal_vif=removed_lookup.get(variable,np.nan)))
        pd.DataFrame(diagnostics).to_csv(folder/'combined_vif_diagnostics.csv',index=False)
        if make_plots:
            plotter = Plotter(folder / 'figures')
            plotter.vif(history[0], self.threshold, 'initial_vif')
            plotter.vif(calculate_vif(data, current), self.threshold, 'final_vif')
        return current
