import pandas as pd
import os

from src.clustering import CountyClustering
from src.data_loader.coordinatesloader import CoordinateLoader
from src.data_loader.dataloader import DataLoader
from src.distance_calculator import DistanceCalculator
from src.eba import ExtremeBoundsAnalyzer
from src.gravity.gravity_calculator import GravityModel
from src.gravity.gravity_migration_model import MigrationGravityModel
from src.plotter import Plotter


class Runner:
    def __init__(self, hub: str = "Nairobi", alpha: float = 0.05,
                 output_dir: str = "output"):
        self.hub = hub
        self.alpha = alpha
        self.output_dir = output_dir

        self.data_any = DataLoader()
        self.data_male = DataLoader(gender="male")
        self.data_fem = DataLoader(gender="female")

        self.coords_loader = CoordinateLoader()
        self.coordinates = self.coords_loader.get_all_coordinates()
        self.distance_calc = DistanceCalculator(self.coordinates, method="geodesic")
        self.distance_calc.compute_all_distances()
        self.dist_from_hub = self.distance_calc.get_distances_from_hub(self.hub)

        self.gdf = self.data_any.load_county_shapefile()
        self.plotter = Plotter(gravity_dict={})
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self) -> None:
        print("\n=== Dataset loaded ===")
        print(f"Counties: {len(self.data_any.get_all_counties())}")
        print(f"OD pairs in distance matrix: {len(self.distance_calc.get_all_distances())}")

        for gender, dl in (("female", self.data_fem), ("male", self.data_male)):
            print(f"\n========== {gender.upper()} ==========")

            shares = dl.data_df["Born_in_Kenya_but_outside"]
            population = dl.data_df["Population"]

            # Gravity
            gm = GravityModel(
                county_emigrant_share=shares,
                population=population,
                distance_calculator=self.distance_calc,
                distance_exponent=2.0,
            )
            gm.compute_gravity()

            # Save full gravity table
            g_all = gm.get_all_gravity()
            df_g = pd.DataFrame(((i, j, v) for (i, j), v in g_all.items()),
                                columns=["origin", "destination", "gravity"])
            df_g.to_csv(os.path.join(self.output_dir, f"gravity_flows_{gender}.csv"),
                        index=False)

            # Top-10 from hub
            flows_from_hub = [((i, j), val) for (i, j), val in g_all.items() if
                              i == self.hub]
            flows_from_hub.sort(key=lambda x: x[1], reverse=True)
            print(f"\nTop 10 gravity flows FROM {self.hub} ({gender}):")
            for (i, j), val in flows_from_hub[:10]:
                print(f"{i} → {j}: {val:,.2f}")

            mig = MigrationGravityModel(
                data_loader=dl,
                distances_from_county_hub=self.dist_from_hub,
                alpha=self.alpha,
                output_dir=self.output_dir,
                target_variable="Born_in_Kenya_but_outside",
            )
            model_full, model_final, selected, df_model = mig.run_model(
                target_dict=shares, label=f"{gender}_{self.hub}"
            )

            if df_model is None or len(selected) == 0:
                print(f"No selected variables for {gender} — skipping EBA.")
            else:
                # Define EBA sets
                # y is log(target_variable)
                y_col = f"log_{mig.target_variable}"

                # FREE variables: gravity core terms you always include
                free_vars = []
                if "log_distance" in df_model.columns:
                    free_vars.append("log_distance")
                if "log_population" in df_model.columns:
                    free_vars.append("log_population")  # keep if you want it always on

                # FOCUS variables: those selected by backward elimination (excluding free)
                focus_vars = [v for v in selected if v not in free_vars]

                # DOUBTFUL pool: everything else available (minus y/free/focus)
                all_X = [c for c in df_model.columns if c != y_col]
                doubtful_pool = [v for v in all_X if v not in set(free_vars + focus_vars)]

                # Run EBA (both Leamer and Sala-i-Martin)
                eba = ExtremeBoundsAnalyzer(
                    df=df_model,
                    y_col=y_col,
                    free_vars=free_vars,
                    focus_vars=focus_vars,
                    doubtful_pool=doubtful_pool,
                    output_dir=os.path.join(self.output_dir, "eba", gender),
                    max_k=3,  # up to 3 doubtful per spec (tune as needed)
                    robust=True,  # HC1 robust SEs
                    weight_by="adj_r2",  # weight Sala-i-Martin by adj. R^2
                    random_subsample=None,  # or e.g. 500 if too many combinations
                    alpha=self.alpha,
                )
                leamer_df, sim_df = eba.analyze()

                # Print concise EBA summaries to console
                print(f"\nEBA — Leamer (stringent) summary [{gender}]")
                if not leamer_df.empty:
                    for _, r in leamer_df.iterrows():
                        print(f"  {r['variable']:>28s} : "
                              f"lower={r['lower_95']:+.4f}  upper={r['upper_95']:+.4f}  "
                              f"robust={bool(r['leamer_robust'])}  (n={int(r['n_specs'])})")
                else:
                    print("  No EBA rows.")

                print(f"\nEBA — Sala-i-Martin summary [{gender}] (weighted by {eba.weight_by})")
                if not sim_df.empty:
                    for _, r in sim_df.iterrows():
                        print(f"  {r['variable']:>28s} : "
                              f"beta_bar={r['beta_bar']:+.4f}  se={r['se_beta']:.4f}  "
                              f"CDF(0)={r['CDF0']:.3f}  (n={int(r['n_specs'])})")
                else:
                    print("  No EBA rows.")

            # Print clusters for BOTH modes
            for mode in ("percent", "count"):
                CountyClustering(
                    immigrants=shares,
                    output_dir=f"{self.output_dir}/clusters",
                    gender=gender,
                ).cluster_on_immigration_data(
                    cluster_threshold=2.3, show_clusters=True,
                    mode=mode,
                    population=population if mode == "count" else None
                )
        self.plot()

    def plot(self) -> None:
        """Create all figures, including percent & count combined pages per gender."""
        for gender, dl in (("female", self.data_fem), ("male", self.data_male)):
            # percent page
            self.plotter.plot_combined_born_outside_view_for_gender(
                shapefile_gdf=self.gdf,
                data_loader=dl,
                distance_from_mandera=self.dist_from_hub,
                gender=gender,
                cluster_threshold=1.8 if gender == "female" else 2.3,
                save_path=os.path.join(self.output_dir,
                                       f"combined_born_outside_{gender}_percent.pdf"),
                shapefile_name_col="NAME",
                mode="percent",
            )
            # count page
            self.plotter.plot_combined_born_outside_view_for_gender(
                shapefile_gdf=self.gdf,
                data_loader=dl,
                distance_from_mandera=self.dist_from_hub,
                gender=gender,
                cluster_threshold=1.5 if gender == "female" else 1.5,
                save_path=os.path.join(self.output_dir,
                                       f"combined_born_outside_{gender}_count.pdf"),
                shapefile_name_col="NAME",
                mode="count",
            )

        # single overlay (per-gender bubbles, distance bands)
        self.plotter.plot_migration_vector_map_gender_overlay(
            distances_from_hub=self.dist_from_hub,
            share_series_male=self.data_male.data_df["Born_in_Kenya_but_outside"],
            share_series_female=self.data_fem.data_df["Born_in_Kenya_but_outside"],
            population_series=self.data_any.data_df["Population"],
            coordinates_dict=self.coordinates, shapefile_gdf=self.gdf,
            output_path=self.output_dir, hub_name=self.hub,
            filename=f"migration_overlay_{self.hub.lower()}.png", )

        # Distance heatmap + correlation matrix
        self.plotter.plot_distance_heatmap(
            self.distance_calc.get_all_distances(),
            self.data_any.get_all_counties(),
        )
        self.plotter.plot_variable_correlation_matrix(
            data=self.data_any,
            output_path=f"{self.output_dir}/correlation_matrix.pdf",
        )

        # Stacked percent bars for both genders
        self.plotter.plot_stacked_percent_bars_by_county(
            data_loader=self.data_any, gender="female",
            filename="county_stacked_selected_vars_female.pdf",
        )
        self.plotter.plot_stacked_percent_bars_by_county(
            data_loader=self.data_any, gender="male",
            filename="county_stacked_selected_vars_male.pdf",
        )


