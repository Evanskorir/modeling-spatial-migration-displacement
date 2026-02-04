import os
import pandas as pd
import geopandas as gpd
from typing import Dict, Any


class DataLoader:

    file_by_gender = {
        "male": "data_male.xls",
        "female": "data_female.xls",
    }

    # Map original column headers
    COLUMN_ALIAS: Dict[str, str] = {
        # migration origin
        # "Born_in_current_place":                "born_in_current_place",
        "Born_in_Kenya_but_outside":            "born_in_kenya_outside_current_place",
        # "Born_outside_Kenya":                   "born_outside_kenya",

        # migration reasons
        "Employment_Migration":                   "reason_employment",
        "Education_Migration":                   "reason_education_training",
        "Marriage_Migration":                     "reason_marriage_formation",
        "Family_Migration":                        "reason_family_reunification",
        "Forced_Migration":                         "reason_forced_displacement",
        "Other_reason_Migration":                   "reason_other",

        # education attainment
        "No_education":                         "edu_none",
        "Some_primary":                         "edu_some_primary",
        "Completed_primary":                    "edu_completed_primary",
        "Some_secondary":                       "edu_some_secondary",
        "Completed_secondary":                  "edu_completed_secondary",
        "More_than_secondary":                  "edu_more_than_secondary",

        # employment status
        "Currently_employed":                   "employed_currently",
        "Not_currently_employed":               "employed_not_currently",
        "Not_employed_last_12_months":          "employed_none_last12m",

        # food security
        "Poor_Food_Consumption_Score":                   "food_poor_pct",
        "Borderline_Food_Consumption_Score":             "food_borderline_pct",
        "Acceptable_Food_Consumption_Score":            "food_acceptable_pct",
        "Mean_Coping_Strategy_Index":           "food_mean_coping_strategy_index",

        # inequality
        "Gini_coefficient":                     "gini_coefficient",

        # access / services
        "Distance_to_health_facility":      "distance_to_health_facility_pct",
        "Water_Availability":                   "water_available_pct",
        "No_Mass_Media_Access":                "media_access_none_pct",
        "Mass_Media_Access":                 "media_access_any_pct",
        "Daily_Internet_Access":            "internet_almost_every_day_pct",
        "Weekly_Internet_access":              "internet_at_least_weekly_pct",

        # geography & economy
        "Population":                           "population",
        "Land_Area":                            "land_area_km2",
        "Population_Density":                   "population_density",
        "Working_Population":                   "working_population",
        "Poverty_Rate":                         "poverty_rate_pct",
        "Gross_Domestic_Product":                "gdp",
        "Average_Household_Size":               "avg_household_size",
    }

    def __init__(self, gender: str = "male"):
        gender = gender.strip().lower()
        if gender not in self.file_by_gender:
            raise ValueError(f"gender must be one of {list(self.file_by_gender)}, "
                             f"got {gender!r}")
        self.gender = gender

        base_dir = os.path.dirname(os.path.realpath(__file__))
        self._county_data_file = os.path.join(base_dir, "../../data",
                                              self.file_by_gender[gender])
        self._geojson_file = os.path.join(base_dir, "../../data",
                                          "kenya_counties.geojson")

        self._load_county_data()

    def _load_county_data(self):
        if not os.path.exists(self._county_data_file):
            raise FileNotFoundError(f"Excel file not found: {self._county_data_file}")

        df = pd.read_excel(self._county_data_file)

        # Normalize column names from Excel
        def norm(c: str) -> str:
            return (
                c.strip()
                 .replace(" ", "_")
                 .replace("(", "")
                 .replace(")", "")
                 .replace("%", "pct")
            )

        df.columns = [norm(c) for c in df.columns]
        df.rename(columns=lambda x: x.replace("__", "_"), inplace=True)

        # Detect & standardize county column
        county_col = next((c for c in df.columns if c.lower() in {
            "county", "counties", "county_name"}), None)
        if not county_col:
            raise KeyError("County column not found in county Excel file.")
        df[county_col] = df[county_col].astype(str).str.strip().str.title()
        df.set_index(county_col, inplace=True)

        # keep numerics where possible
        df = df.apply(pd.to_numeric, errors="ignore")
        self.data_df = df

        # Build a dict of variables with clean names -> dict(county -> value)
        self.variables: Dict[str, Dict[str, Any]] = {}
        for raw_col in df.columns:
            clean_key = self.COLUMN_ALIAS.get(raw_col, raw_col.lower())
            self.variables[clean_key] = df[raw_col].to_dict()

        # Also expose each variable as an attribute for convenience
        for name, mapping in self.variables.items():
            setattr(self, name, mapping)

    def get_all_data(self) -> pd.DataFrame:
        return self.data_df

    def get_all_counties(self):
        return list(self.data_df.index)

    def get_variables(self) -> Dict[str, Dict[str, Any]]:
        """Returns {variable_name: {county: value}} for all variables."""
        return self.variables

    def get_value_for_county(self, county: str, variable: str):
        county = county.strip().title()
        if variable in self.variables and county in self.variables[variable]:
            return self.variables[variable][county]
        return None

    def load_county_shapefile(self) -> gpd.GeoDataFrame:
        if not os.path.exists(self._geojson_file):
            raise FileNotFoundError(f"GeoJSON file not found at: {self._geojson_file}")
        gdf = gpd.read_file(self._geojson_file)
        if "NAME_1" in gdf.columns:
            gdf["NAME"] = gdf["NAME_1"].str.title()
        elif "NAME" in gdf.columns:
            gdf["NAME"] = gdf["NAME"].str.title()
        else:
            raise ValueError("County name column not found in shapefile.")
        return gdf



