import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm

from matplotlib.backends.backend_pdf import PdfPages


class SpatialGravityModel:
    def __init__(self, data_loader, distance_matrix, distances_from_county_hub,
                 target_variable, alpha=0.05, output_dir="output"):
        self.data = data_loader
        self.distance_matrix = distance_matrix
        self.distances = distances_from_county_hub
        self.alpha = alpha
        self.output_dir = output_dir
        self.target_variable = target_variable
        os.makedirs(output_dir, exist_ok=True)

    def _build_spatial_lag(self, log_target_series):
        counties = log_target_series.index.tolist()
        W = pd.DataFrame(0.0, index=counties, columns=counties)
        for i in counties:
            weights = {}
            total_weight = 0.0
            for j in counties:
                if i == j:
                    continue
                d = self.distance_matrix.get((i, j)) or self.distance_matrix.get((j, i))
                if d and d > 0:
                    w = 1.0 / d
                    weights[j] = w
                    total_weight += w

            for j, w in weights.items():
                W.loc[i, j] = w / total_weight if total_weight > 0 else 0.0

        spatial_lag = W.values @ log_target_series.values
        return pd.Series(spatial_lag, index=counties, name="spatial_lag")

    def _get_feature_value(self, county, var, distance):
        d = self.data
        if var.startswith("log_"):
            base = var.replace("log_", "")
            if base == "distance":
                return np.log(distance + 1e-6)

            value = getattr(d, base, None)
            if value and county in value:
                return np.log(value[county] + 1e-6)
        # linear / fractional variables
        value = getattr(d, var, None)
        if value and county in value:
            v = value[county]
            return v / 100.0 if v > 1 else v
        raise KeyError(f"Variable '{var}' not found for county {county}")

    def fit_model_for_single_point(self, label, selected_vars, target_dict):
        print(f"Fitting SAR model for {label}...")
        records = []
        for county, distance in self.distances.items():
            N = target_dict.get(county)
            if N is None or N <= 0:
                continue
            record = {
                "county": county,
                f"log_{self.target_variable}": np.log(N),
            }

            try:
                for var in selected_vars:
                    record[var] = self._get_feature_value(county, var, distance)
                records.append(record)
            except KeyError:
                continue
        df = pd.DataFrame(records).set_index("county")
        log_col = f"log_{self.target_variable}"
        df["spatial_lag"] = self._build_spatial_lag(df[log_col])
        X = df[selected_vars + ["spatial_lag"]]
        X = sm.add_constant(X)
        y = df[log_col]

        model = sm.OLS(y, X).fit()
        self._save_summary_as_pdf(model, f"sar_summary_{label}.pdf")
        return model

    def _save_summary_as_pdf(self, model, filename):
        summary_text = model.summary().as_text()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        ax.text(0, 1, summary_text, fontsize=9, fontfamily="monospace", va="top")

        path = os.path.join(self.output_dir, filename)
        with PdfPages(path) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def fit_all_models(self, selected_vars_by_label, target_dict_by_label):
        for label, selected_vars in selected_vars_by_label.items():
            target_dict = target_dict_by_label.get(label)
            if target_dict:
                self.fit_model_for_single_point(label, selected_vars, target_dict)
            else:
                print(f"Missing data for {label}")
