"""Read the existing workbooks without editing or imputing source values."""
from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd


OUTCOME = "Born_in_Kenya_but_outside"
PERCENT_COLUMNS = {
    OUTCOME, "Born_in_current_place", "Born_outside_Kenya",
    "Employment_Migration", "Education_Migration", "Marriage_Migration",
    "Family_Migration", "Forced_Migration", "Other_reason_Migration",
    "No_education", "Some_primary", "Completed_primary", "Some_secondary",
    "Completed_secondary", "More_than_secondary", "Currently_employed",
    "Not_currently_employed", "Not_employed_last_12_months",
    "Poor_Food_Consumption_Score", "Borderline_Food_Consumption_Score",
    "Acceptable_Food_Consumption_Score", "Distance to health facility",
    "Water_Availability", "No_Mass_Media_Access", "Mass_Media_Access",
    "Daily_Internet_Access", "Weekly_Internet_access", "Poverty Rate",
}
COMPOSITIONS = {
    "education": ["No_education", "Some_primary", "Completed_primary",
                  "Some_secondary", "Completed_secondary", "More_than_secondary"],
    "employment": ["Currently_employed", "Not_currently_employed",
                   "Not_employed_last_12_months"],
    "food_consumption": ["Poor_Food_Consumption_Score",
                         "Borderline_Food_Consumption_Score",
                         "Acceptable_Food_Consumption_Score"],
    "media_access": ["No_Mass_Media_Access", "Mass_Media_Access"],
    "migration_reasons": ["Employment_Migration", "Education_Migration",
                          "Marriage_Migration", "Family_Migration",
                          "Forced_Migration", "Other_reason_Migration"],
    "birthplace": ["Born_in_current_place", OUTCOME, "Born_outside_Kenya"],
}


def county_key(value):
    """Join key only; retain the workbook's county label for display."""
    key = re.sub(r"[^a-z0-9]", "", str(value).casefold())
    return {"nairobicity": "nairobi", "elgeyomarakwet": "elgeyomarakwet"}.get(key, key)


class MigrationDataLoader:
    """Validate a matched pair of county workbooks; no geographic dependencies.

    Input percentages MUST use the supplied 0--100 scale. Numeric source values
    are retained. Only whitespace in headers/labels and join keys is normalized.
    A fatal outcome/schema problem stops the run; other issues are reported.
    """

    def __init__(self, data_dir, female_file="data_female.xls",
                 male_file="data_male.xls", expected_counties=47):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.files = {"female": self.data_dir / female_file,
                      "male": self.data_dir / male_file}
        self.expected_counties = expected_counties
        self.frames = {}
        self.source_manifest = []
        self.issues = []
        self.composition_checks = []
        for sex, path in self.files.items():
            self.frames[sex] = self._read(path, sex)
        self._validate_pair()

    def _issue(self, sex, county, variable, value, issue):
        self.issues.append(dict(sex=sex, county=county, variable=variable,
                                value=value, issue=issue))

    def _read(self, path, sex):
        if not path.is_file():
            raise FileNotFoundError(f"Workbook not found: {path}")
        df = pd.read_excel(path, sheet_name=0)
        df.columns = [str(c).strip() for c in df.columns]
        if df.columns.duplicated().any():
            raise ValueError(f"Duplicate column headers in {path.name}")
        if "County" not in df or OUTCOME not in df:
            raise ValueError(f"{path.name} must contain County and {OUTCOME}")
        if df.County.isna().any() or df.County.astype(str).str.strip().eq("").any():
            raise ValueError(f"Blank county label in {path.name}")
        df["County"] = df.County.astype(str).str.strip()
        df.insert(1, "county_key", df.County.map(county_key))
        if df.county_key.duplicated().any():
            raise ValueError(f"Duplicate county join keys in {path.name}")
        if len(df) != self.expected_counties:
            raise ValueError(f"{path.name}: expected {self.expected_counties} counties, got {len(df)}")
        for col in df.columns.difference(["County", "county_key"]):
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Nonnumeric value in {path.name}, column {col}; inspect source") from exc
            for i in df.index:
                v = df.at[i, col]
                if pd.isna(v) or not np.isfinite(v):
                    self._issue(sex, df.at[i, "County"], col, v, "missing_or_nonfinite")
                elif v < 0 or (col in PERCENT_COLUMNS and v > 100):
                    self._issue(sex, df.at[i, "County"], col, v, "outside_expected_range")
        y = df[OUTCOME]
        if not np.isfinite(y).all() or not y.between(0, 100).all():
            raise ValueError(f"{path.name}: outcome must be finite and within 0--100; no rows dropped")
        for name, cols in COMPOSITIONS.items():
            if not set(cols).issubset(df):
                continue
            # Rounding tolerance for k percentages each reported to one decimal.
            tolerance = 0.05 * len(cols) + 1e-8
            totals = df[cols].sum(axis=1, min_count=len(cols))
            for i, total in totals.items():
                complete = np.isfinite(df.loc[i, cols].to_numpy(dtype=float)).all()
                status = ("missing_or_nonfinite_component" if not complete else
                          "within_rounding_tolerance" if abs(total - 100) <= tolerance else
                          "all_zero_check_denominator_or_suppression" if total == 0 else
                          "sum_not_100_check_definition_or_source")
                self.composition_checks.append(dict(
                    sex=sex, county=df.at[i, "County"], group=name,
                    total_pct=total, rounding_tolerance_pp=tolerance, status=status))
                if status != "within_rounding_tolerance":
                    self._issue(sex, df.at[i, "County"], name, total, status)
        self.source_manifest.append(dict(
            sex=sex, source_file=str(path), rows=len(df),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return df

    def _validate_pair(self):
        female, male = self.frames["female"], self.frames["male"]
        if set(female.county_key) != set(male.county_key):
            raise ValueError("Female and male county sets differ; no automatic dropping or imputation")
        if set(female.columns) != set(male.columns):
            raise ValueError("Female and male workbook schemas differ; select a matched pair of files")

    def get_data(self, sex):
        return self.frames[sex].copy(deep=True)

    def get_long_data(self):
        return pd.concat([df.assign(sex=sex) for sex, df in self.frames.items()],
                         ignore_index=True)

    def get_paired_outcomes(self):
        left = self.frames["female"][["County", "county_key", OUTCOME]].rename(
            columns={OUTCOME: "female_pct"})
        right = self.frames["male"][["county_key", OUTCOME]].rename(
            columns={OUTCOME: "male_pct"})
        df = left.merge(right, on="county_key", validate="one_to_one")
        df["gap_pp"] = df.female_pct - df.male_pct
        # Inputs are rounded to 0.1 pp; tolerance only protects exact equality.
        df["higher_recorded_prevalence"] = np.select(
            [df.gap_pp > 1e-9, df.gap_pp < -1e-9], ["female", "male"], default="equal")
        return df.sort_values(["gap_pp", "County"], ascending=[False, True]).reset_index(drop=True)
