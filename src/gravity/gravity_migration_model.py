import numpy as np
from src.gravity.gravity_base_model import BaseGravityModel


class MigrationGravityModel(BaseGravityModel):
    def __init__(self, data_loader, distances_from_county_hub, alpha=0.05,
                 output_dir="output", target_variable="migration"):

        features = [
            # logged scales
            "log_population", "log_working", "log_gdp",
            "log_pop_density", "log_land_area", "log_distance",
            # linear scales
            "avg_household_size", "gini",
            # access / services (fractions)
            "poverty_rate", "media_access_any", "media_access_none",
            "internet_almost_daily", "internet_weekly",
            "distance_to_health_facility", "water_available",
            # migration reasons (fractions)
            "reason_employment", "reason_education_training",
            "reason_marriage_formation", "reason_family_reunification",
            "reason_forced_displacement", "reason_other",
            # education attainment (fractions)
            "edu_none", "edu_some_primary", "edu_completed_primary",
            "edu_some_secondary", "edu_completed_secondary", "edu_more_than_secondary",
            # employment status (fractions)
            "employed_currently", "employed_not_currently", "employed_none_last12m",
            # food security (fractions) + CSI linear
            "food_poor", "food_borderline", "food_acceptable", "food_csi",
        ]

        super().__init__(data_loader, distances_from_county_hub, target_variable,
                         features, alpha, output_dir)

    def _extract_features(self, county, distance, _):
        # Core scales (log)
        P = self.data.population[county]
        W = self.data.working_population[county]
        G = self.data.gdp[county]
        PD = self.data.population_density[county]
        LA = self.data.land_area_km2[county]

        # Linear scales
        AHS = self.data.avg_household_size[county]
        GC = self.data.gini_coefficient[county]

        # Fractions: convert 0–100 → 0–1
        PR = self.data.poverty_rate_pct[county] / 100.0
        MMN = self.data.media_access_none_pct[county] / 100.0
        MMA = self.data.media_access_any_pct[county] / 100.0
        IAD = self.data.internet_almost_every_day_pct[county] / 100.0
        IW = self.data.internet_at_least_weekly_pct[county] / 100.0
        DTH = self.data.distance_to_health_facility_pct[county] / 100.0
        WAT = self.data.water_available_pct[county] / 100.0

        RE = self.data.reason_employment[county] / 100.0
        RET = self.data.reason_education_training[county] / 100.0
        RMF = self.data.reason_marriage_formation[county] / 100.0
        RFR = self.data.reason_family_reunification[county] / 100.0
        RFD = self.data.reason_forced_displacement[county] / 100.0
        RO = self.data.reason_other[county] / 100.0

        E0 = self.data.edu_none[county] / 100.0
        E1 = self.data.edu_some_primary[county] / 100.0
        E2 = self.data.edu_completed_primary[county] / 100.0
        E3 = self.data.edu_some_secondary[county] / 100.0
        E4 = self.data.edu_completed_secondary[county] / 100.0
        E5 = self.data.edu_more_than_secondary[county] / 100.0

        EC = self.data.employed_currently[county] / 100.0
        ENC = self.data.employed_not_currently[county] / 100.0
        E12 = self.data.employed_none_last12m[county] / 100.0

        FP = self.data.food_poor_pct[county] / 100.0
        FB = self.data.food_borderline_pct[county] / 100.0
        FA = self.data.food_acceptable_pct[county] / 100.0
        CSI = self.data.food_mean_coping_strategy_index[county]

        return {
            # logged
            "log_population":     np.log(P + 1e-6),
            "log_working":        np.log(W + 1e-6),
            "log_gdp":            np.log(G + 1e-6),
            "log_pop_density":    np.log(PD + 1e-6),
            "log_land_area":      np.log(LA + 1e-6),
            "log_distance":       np.log(distance + 1e-6),
            # linear
            "avg_household_size": AHS,
            "gini":               (GC / 100.0) if GC > 1 else GC,
            # access/services
            "poverty_rate": PR,
            "media_access_any": MMA,
            "media_access_none": MMN,
            "internet_almost_daily": IAD,
            "internet_weekly": IW,
            "distance_to_health_facility": DTH,
            "water_available": WAT,
            # migration reasons
            "reason_employment": RE,
            "reason_education_training": RET,
            "reason_marriage_formation": RMF,
            "reason_family_reunification": RFR,
            "reason_forced_displacement": RFD,
            "reason_other": RO,
            # education attainment
            "edu_none": E0,
            "edu_some_primary": E1,
            "edu_completed_primary": E2,
            "edu_some_secondary": E3,
            "edu_completed_secondary": E4,
            "edu_more_than_secondary": E5,
            # employment status
            "employed_currently": EC,
            "employed_not_currently": ENC,
            "employed_none_last12m": E12,
            # food security
            "food_poor": FP,
            "food_borderline": FB,
            "food_acceptable": FA,
            "food_csi": CSI,
        }

