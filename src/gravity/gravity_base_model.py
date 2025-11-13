import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm

from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use("Agg")


class BaseGravityModel:
    def __init__(self, data_loader, distances_from_county_hub,
                 target_variable, features, alpha=0.05, output_dir="output"):
        self.data = data_loader
        self.distances = distances_from_county_hub
        self.target_variable = target_variable
        self.features = features
        self.alpha = alpha
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _extract_features(self, county, distance, raw_target):
        raise NotImplementedError("Subclasses must implement this method")

    def _prepare_data(self, target_dict):
        records = []
        skipped = []

        for county, distance in self.distances.items():
            if county in ["Nairobi", "Mombasa"]:
                continue

            try:
                target_value = target_dict.get(county)
                if target_value is None or target_value <= 0:
                    skipped.append(county)
                    continue

                feature_record = self._extract_features(county, distance, target_value)
                if feature_record:
                    feature_record[f"log_{self.target_variable}"] = np.log(target_value)
                    records.append(feature_record)
                else:
                    skipped.append(county)

            except Exception:
                skipped.append(county)

        return pd.DataFrame(records)

    def _backward_elimination(self, X, y):
        X = sm.add_constant(X)
        while True:
            model = sm.OLS(y, X).fit()
            pvals = model.pvalues.drop("const", errors="ignore")
            if pvals.empty:
                break
            max_pval = pvals.max()
            if max_pval > self.alpha:
                X = X.drop(columns=[pvals.idxmax()])
            else:
                break
        return X.columns.tolist(), model

    def _save_summary_as_pdf(self, model, filename):
        summary_text = model.summary().as_text()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        ax.text(0, 1, "\n".join(summary_text.splitlines()), fontsize=9,
                fontfamily='monospace', va='top')
        path = os.path.join(self.output_dir, filename)
        with PdfPages(path) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    def run_model(self, target_dict, label=""):
        print(f"Fitting gravity model for {label}...")
        df = self._prepare_data(target_dict)
        self._last_df = df.copy()  # <--- store for EBA access

        if df.empty:
            print(f"Skipping {label} — no valid data.")
            return None, None, [], df  # <--- also return df

        y = df[f"log_{self.target_variable}"]
        X_all = df.drop(columns=[f"log_{self.target_variable}"])

        model_full = sm.OLS(y, sm.add_constant(X_all)).fit()
        self._save_summary_as_pdf(model_full, f"ols_full_{label}.pdf")

        selected_features, _ = self._backward_elimination(X_all, y)
        selected = [f for f in selected_features if f != "const"]
        model_final = sm.OLS(y, sm.add_constant(df[selected])).fit()
        self._save_summary_as_pdf(model_final, f"ols_selected_{label}.pdf")

        return model_full, model_final, selected, df



