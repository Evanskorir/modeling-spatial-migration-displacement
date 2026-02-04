import matplotlib
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from collections import defaultdict
from matplotlib import gridspec, patheffects
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram, fcluster
from src.clustering import CountyClustering
matplotlib.use("Agg")


class Plotter:
    def __init__(self, gravity_dict, output_dir="output"):

        self.dendo_colors = None
        self.gravity_dict = gravity_dict
        self.output_dir = output_dir
        self.gravity_df = self._to_dataframe()

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # --- Region Definitions ---
        self.region_order = {
            "Nairobi": ["Nairobi"],
            "Coast": ["Mombasa", "Kwale", "Kilifi", "Tana River", "Lamu", "Taita Taveta"],
            "North Eastern": ["Garissa", "Wajir", "Mandera"],
            "Eastern": ["Kitui", "Machakos", "Makueni", "Meru", "Embu", "Tharaka Nithi",
                "Isiolo", "Marsabit"],
            "Central": ["Muranga", "Kiambu", "Nyandarua", "Nyeri", "Kirinyaga"],
            "Rift Valley": ["Baringo", "Bomet", "Elgeyo Marakwet", "Kajiado", "Kericho",
                "Laikipia", "Nakuru", "Nandi", "Narok", "Samburu", "Trans Nzoia",
                "Turkana", "Uasin Gishu", "West Pokot"],
            "Western": ["Bungoma", "Busia", "Kakamega", "Vihiga"],
            "Nyanza": ["Homa Bay", "Kisii", "Kisumu", "Migori", "Nyamira", "Siaya"]
        }
        self.correction_map = {
            "Homabay": "Homa Bay", "HomaBay": "Homa Bay",
            "Murang'A": "Muranga",
            "Taitataveta": "Taita Taveta",
            "Tanariver": "Tana River", "Transnzoia": "Trans Nzoia",
            "Uasingishu": "Uasin Gishu", "Westpokot": "West Pokot"
        }

    def _canonical(self, name: str) -> str:
        """Robust county name canonicalization."""
        if name is None:
            return name
        # normalize punctuation/spacing/case
        n = str(name).strip()
        n = n.replace("–", "-").replace("—", "-")
        n = n.replace("’", "'").replace("‘", "'")
        n = n.replace("_", " ").replace("-", " ")
        n = " ".join(n.split())  # collapse multiple spaces
        n = n.title()

        # map known variants
        alias = {
            "Homabay": "Homa Bay",  # <- the one biting you
            "HomaBay": "Homa Bay",
            "Murang'A": "Muranga",
            "Taitataveta": "Taita Taveta",
            "Tanariver": "Tana River",
            "Transnzoia": "Trans Nzoia",
            "Uasingishu": "Uasin Gishu",
            "Westpokot": "West Pokot",
        }
        return alias.get(n, n)

    def _norm_series_index(self, s: pd.Series) -> pd.Series:
        """Normalize index names and collapse duplicates by summing."""
        s2 = s.copy()
        s2.index = [self._canonical(i) for i in s2.index]
        # If the source had both "Homabay" and "Homa Bay", sum them.
        return s2.groupby(level=0).sum()

    def _norm_dict_keys(self, d: dict) -> dict:
        """Normalize dict keys (1-level) like {'Homabay': 123}."""
        from collections import defaultdict
        out = defaultdict(float)
        for k, v in d.items():
            out[self._canonical(k)] += v
        return dict(out)

    def _norm_coord_dict(self, d: dict) -> dict:
        """Normalize coordinate dict keys, keep first non-null."""
        out = {}
        for k, v in d.items():
            ck = self._canonical(k)
            if ck not in out or out[ck] in (None, (None, None)):
                out[ck] = v
        return out

    @staticmethod
    def normalize_county_name(name):
        name = name.lower().replace("-", " ").replace("_", " ")
        name = name.replace("’", "'").replace("‘", "'")
        name = name.replace("'", "")  # keep ONLY this line, drop .replace("", "'")
        name = " ".join(word.capitalize() for word in name.split())
        return name

    def _to_dataframe(self):
        # Normalize all county names in the keys
        normalized_gravity = {
            (self.normalize_county_name(i), self.normalize_county_name(j)): val
            for (i, j), val in self.gravity_dict.items()
        }

        counties = sorted(set([i for i, _ in normalized_gravity.keys()] +
                              [j for _, j in normalized_gravity.keys()]))

        df = pd.DataFrame(index=counties, columns=counties)
        for (origin, dest), value in normalized_gravity.items():
            df.loc[origin, dest] = value
        return df.astype(float)

    @staticmethod
    def convert_distance_dict_to_df(distance_dict, counties):
        """
        Converts a pairwise distance dictionary to a square DataFrame.
        """
        df = pd.DataFrame(index=counties, columns=counties)

        for (origin, dest), val in distance_dict.items():
            df.at[origin, dest] = val

        # Fill diagonals with 0 and convert to float
        np.fill_diagonal(df.values, 0)
        return df.astype(float)

    def plot_distance_heatmap(self, distance_dict, counties,
                              filename="distance_matrix_heatmap.pdf"):
        """
        Plots a square heatmap of county-to-county distances using Matplotlib with
        annotations.
        """
        distance_df = self.convert_distance_dict_to_df(distance_dict, counties)
        values = distance_df.values
        values = distance_df.values[::-1]

        fig, ax = plt.subplots(figsize=(20, 20))

        # Heatmap with colormap
        norm = Normalize(vmin=0, vmax=600)
        im = ax.imshow(values, cmap="viridis", norm=norm)

        # Set axis labels and ticks
        ax.set_xticks(np.arange(len(counties)))
        ax.set_yticks(np.arange(len(counties)))
        ax.set_xticklabels(counties, rotation=90, fontsize=18)
        ax.set_yticklabels(counties[::-1], fontsize=18)

        ax.set_xlabel("County of origin", fontsize=25, labelpad=15)
        ax.set_ylabel("Destination County", fontsize=25, labelpad=15)

        ax.tick_params(which="minor", bottom=False, left=False)

        # Axis border visibility
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)

        # Colorbar setup
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.4)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label("Distance (km)", fontsize=24)
        cb.set_ticks([100, 200, 300, 400, 500, 600])
        cb.ax.tick_params(labelsize=25)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=400)
        plt.close()
        print(f"Distance matrix heatmap saved to {save_path}")

    @staticmethod
    def plot_gender_correlation_difference(
            data_male,
            data_female,
            output_path="output/correlation_matrix_gender_difference.pdf",
    ):
        """
        Plot Male − Female correlation matrix difference,
        visually aligned with plot_variable_correlation_matrix,
        using a colorbar instead of annotations.
        """

        # --- Load raw data ---
        df_m = data_male.get_all_data()
        df_f = data_female.get_all_data()

        # --- SAME correlation logic as individual plots ---
        corr_m = df_m.corr().round(2)
        corr_f = df_f.corr().round(2)

        # --- Enforce identical ordering ---
        corr_f = corr_f.loc[corr_m.index, corr_m.columns]

        # --- Difference ---
        corr_diff = (corr_m - corr_f).round(2)
        corr_diff = corr_diff.iloc[::-1]

        labels = [col.replace("_", " ") for col in corr_diff.columns]

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(16, 14))

        norm = Normalize(vmin=-1, vmax=1)
        im = ax.imshow(corr_diff.values, cmap="coolwarm", norm=norm)

        ticks = np.arange(len(corr_diff.columns))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=14)
        ax.set_yticklabels(labels[::-1], fontsize=14)

        ax.tick_params(bottom=True, top=False, labelbottom=True, labeltop=False)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)

        # --- Colorbar (same pattern as distance heatmap) ---
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.4)

        cb = plt.colorbar(im, cax=cax)
        cb.set_label("Correlation Difference (Male − Female)", fontsize=16)
        cb.ax.tick_params(labelsize=14)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"gender correlation difference saved to {output_path}")

    @staticmethod
    def plot_variable_correlation_matrix(data,
                                         output_path="output/correlation_matrix.pdf"):
        df = data.get_all_data()

        # Define variables
        base_cols = [
            "born_in_kenya_outside_current_place",
            "reason_employment", "reason_education_training",
            "reason_marriage_formation", "reason_family_reunification",
            "reason_forced_displacement", "reason_other", "edu_none",
            "edu_some_primary",  "edu_completed_primary", "edu_some_secondary",
            "edu_completed_secondary",  "edu_more_than_secondary", "employed_currently",
            "employed_not_currently", "employed_none_last12m", "food_poor_pct",
            "food_borderline_pct", "food_acceptable_pct", "food_mean_coping_strategy_index",
            "gini_coefficient",  "distance_to_health_facility_pct", "water_available_pct",
            "media_access_none_pct", "media_access_any_pct",
            "internet_almost_every_day_pct", "internet_at_least_weekly_pct", "population",
            "land_area_km2", "population_density", "working_population",
            "poverty_rate_pct", "gdp", "avg_household_size"

        ]

        col_map = {col.replace(" ", "_"): col for col in base_cols}

        # Final column order
        final_cols = [col.replace(" ", "_") for col in base_cols]
        # df = df[final_cols].dropna()
        corr = df.corr().round(2)
        corr = corr.iloc[::-1]

        # Axis labels
        labels = [col_map.get(col, col.replace("_", " ")) for col in corr.columns]
        fig, ax = plt.subplots(figsize=(16, 14))
        cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

        ticks = np.arange(len(corr.columns))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=14)
        ax.set_yticklabels(labels[::-1], fontsize=14)

        ax.tick_params(bottom=True, top=False, labelbottom=True, labeltop=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)

        # Annotate
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                val = corr.iloc[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.1f}", ha='center', va='center',
                        fontsize=10, color=color, fontweight="bold")

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"correlation matrix saved to {output_path}")

    def plot_migration_vector_map_gender_overlay(
            self,
            distances_from_hub: dict[str, float],
            share_series_male,  # pd.Series: index=county
            share_series_female,  # pd.Series: index=county
            population_series,  # pd.Series: index=county
            coordinates_dict,  # dict OR object with .get_all_coordinates()
            shapefile_gdf,  # GeoDataFrame of county polygons
            output_path: str = "output/figs",
            hub_name: str = "Mandera",
            filename: str = "migration_overlay_mandera_clean.png",
    ):
        if not isinstance(coordinates_dict, dict):
            if hasattr(coordinates_dict, "get_all_coordinates"):
                coordinates_dict = coordinates_dict.get_all_coordinates()
            else:
                raise TypeError("coordinates_dict must be a "
                                "dict or expose get_all_coordinates()")

        # normalize *all* inputs
        share_series_female = self._norm_series_index(share_series_female)
        share_series_male = self._norm_series_index(share_series_male)
        population_series = self._norm_series_index(population_series)

        distances_from_hub = self._norm_dict_keys(distances_from_hub)
        coordinates_dict = self._norm_coord_dict(
            coordinates_dict if isinstance(coordinates_dict, dict)
            else coordinates_dict.get_all_coordinates()
        )
        missing_in_pop = set(distances_from_hub) - set(population_series.index)
        if missing_in_pop:
            print("[warn] names in distances not in population "
                  "after normalization:", missing_in_pop)
        def clean_series(s):
            s = s.astype(str).str.replace(",", ".", regex=False)
            return pd.to_numeric(s, errors="coerce")

        def shares_to_counts(share_s, pop_s):
            s = clean_series(share_s).reindex(pop_s.index)
            p = clean_series(pop_s)
            med = np.nanmedian(s.values) if np.isfinite(np.nanmedian(s.values)) else 0
            scale = 1.0 if med <= 1.5 else 0.01  # proportion vs percent
            return (s * scale * p).fillna(0.0)

        def band_color(d: float):
            # high-contrast palette
            if d < 400:
                return "<400 km", "black"
            if d < 500:
                return "400–500 km", "darkblue"
            if d < 600:
                return "500–600 km", "green"
            if d < 700:
                return "600–700 km", "orange"
            return ">700 km", "red"

        def safe_coord(name):
            if name not in coordinates_dict:
                return None
            lat, lon = coordinates_dict[name]
            return float(lat), float(lon)

        def commas(x: float) -> str:
            return f"{int(max(x, 0)): ,}".replace("\u00a0", ",")

        female_counts = shares_to_counts(share_series_female, population_series)
        male_counts = shares_to_counts(share_series_male, population_series)
        total_counts = female_counts + male_counts

        all_counties = sorted(
            set(population_series.index) |
            set(distances_from_hub.keys()) |
            set(coordinates_dict.keys()) |
            set(female_counts.index) | set(male_counts.index)
        )
        female_counts = female_counts.reindex(all_counties).fillna(0.0)
        male_counts = male_counts.reindex(all_counties).fillna(0.0)
        total_counts = total_counts.reindex(all_counties).fillna(0.0)
        df_counts = pd.DataFrame({"Female": female_counts, "Male": male_counts,
                                  "Total": total_counts})
        os.makedirs(output_path, exist_ok=True)
        csv_path = os.path.join(output_path, "mandera_migrant_counts_by_county.csv")
        df_counts.to_csv(csv_path, float_format="%.0f")

        hub_xy = safe_coord(hub_name)
        if hub_xy is None:
            raise ValueError(f"Hub '{hub_name}' not found in coordinates.")
        y_hub, x_hub = hub_xy

        # ---------- figure ----------
        plt.rcParams.update({
            "font.size": 10,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "font.family": "DejaVu Sans"
        })

        fig, ax = plt.subplots(1, 1, figsize=(16, 16), dpi=300)
        shapefile_gdf.plot(ax=ax, edgecolor="#666666", facecolor="white",
                           linewidth=0.9, zorder=0)

        # hub marker
        ax.plot(x_hub, y_hub, marker="*", color="black", markersize=18, zorder=10)
        ax.text(x_hub, y_hub, hub_name, fontsize=10, ha='right', va='top',
                fontweight='bold', color='black')

        # arrows (line width keyed to total to convey flow salience)
        used_bands = set()
        max_flow = float(max(total_counts.max(), 1.0))

        def flow_lw(count):  # thin: 0.6–1.2 pt
            return 0.6 + 0.6 * np.sqrt(max(count, 0) / max_flow)

        for county, dist in distances_from_hub.items():
            if county == hub_name:
                continue
            coord = safe_coord(county)
            if coord is None:
                continue
            y2, x2 = coord
            band, col = band_color(float(dist))
            used_bands.add(band)

            count = float(total_counts.get(county, 0.0) or 0.0)
            lw = flow_lw(count)
            patch = FancyArrowPatch(
                (x_hub, y_hub), (x2, y2),
                arrowstyle='-|>', color=col,
                linewidth=lw, alpha=0.95,
                connectionstyle="arc3,rad=0.08", capstyle='round',
                zorder=2
            )
            patch.set_path_effects([
                patheffects.Stroke(linewidth=lw + 0.8, foreground='white', alpha=0.9),
                patheffects.Normal()
            ])
            ax.add_patch(patch)

        female_color, male_color = "green", "orange"
        max_gender = float(max(female_counts.max(), male_counts.max(), 1.0))
        k = 0.12
        min_r = 0.045

        def bubble_radius(count):  # per-gender radius
            r = k * np.sqrt(max(count, 0) / max_gender)
            return max(r, min_r)

        def separated_positions(x, y, x0, y0, rf, rm, pad=0.006, sep_factor=0.85):
            dx, dy = x - x0, y - y0
            norm = np.hypot(dx, dy) or 1.0
            px, py = -(dy / norm), (dx / norm)  # perpendicular
            delta = sep_factor * (rf + rm) * 0.5 + pad
            return (x + px * delta, y + py * delta), (x - px * delta, y - py * delta)

        label_texts = []
        for county in all_counties:
            if county == hub_name:
                continue
            coord = safe_coord(county)
            if coord is None:
                continue
            y2, x2 = coord

            fcnt = float(female_counts.get(county, 0.0) or 0.0)
            mcnt = float(male_counts.get(county, 0.0) or 0.0)
            rf, rm = bubble_radius(fcnt), bubble_radius(mcnt)

            (xF, yF), (xM, yM) = separated_positions(x2, y2, x_hub, y_hub, rf, rm)

            if fcnt > 0:
                ax.add_patch(Circle((xF, yF), radius=rf,
                                    facecolor=female_color, edgecolor="black",
                                    lw=0.35, alpha=0.95, zorder=6))
            if mcnt > 0:
                ax.add_patch(Circle((xM, yM), radius=rm,
                                    facecolor=male_color, edgecolor="black",
                                    lw=0.35, alpha=0.95, zorder=6))

            # county label with halo
            txt = ax.text(x2, y2, county, fontsize=8.3, ha='left', va='bottom',
                          color='black', zorder=8)
            txt.set_path_effects([
                patheffects.Stroke(linewidth=2.4, foreground='white'),
                patheffects.Normal()
            ])
            label_texts.append(txt)

        # simple label-repel pass
        def repel_texts(texts, dx=0.02, dy=0.02, iters=120):
            renderer = ax.figure.canvas.get_renderer()
            for _ in range(iters):
                moved = False
                for i in range(len(texts)):
                    if not texts[i].get_visible(): continue
                    xi, yi = texts[i].get_position()
                    bi = texts[i].get_window_extent(renderer=renderer)
                    for j in range(i + 1, len(texts)):
                        if not texts[j].get_visible(): continue
                        bj = texts[j].get_window_extent(renderer=renderer)
                        if bi.overlaps(bj):
                            xj, yj = texts[j].get_position()
                            texts[i].set_position((xi - dx, yi + dy))
                            texts[j].set_position((xj + dx, yj - dy))
                            moved = True
                if not moved:
                    break

        repel_texts(label_texts)

        # legend (distance bands + per-gender size bins + gender)
        band_order = ["<400 km", "400–500 km", "500–600 km", "600–700 km", ">700 km"]
        band_palette = {
            "<400 km": "black",
            "400–500 km": "darkblue",
            "500–600 km": "green",
            "600–700 km": "orange",
            ">700 km": "red",
        }
        band_handles = [Line2D([], [], color=band_palette[b], linewidth=2.2, label=b)
                        for b in band_order if b in used_bands]

        size_edges = [0, 100_000, 250_000, 400_000, 600_000, np.inf]
        size_labels = [
            f"≤ {commas(100_000)}",
            f"{commas(100_000)}–{commas(250_000)}",
            f"{commas(250_000)}–{commas(400_000)}",
            f"{commas(400_000)}–{commas(600_000)}",
            f"> {commas(600_000)}",
        ]

        legend_mids = []
        for lo, hi in zip(size_edges[:-1], size_edges[1:]):
            if np.isinf(hi):
                # pick a midpoint that reflects the upper tail but not the absolute max
                hi = max(1_000_000, 0.6 * max_gender + 600_000)  # ~1.0M when max≈1.8M
            legend_mids.append((lo + hi) / 2.0)

        # convert the per-gender radius (data units) to marker size (points)
        # using a calibrated factor so the dots look comparable to map bubbles
        def legend_markersize(v, ms_factor=180.0, ms_min=6.0):
            r_data = k * np.sqrt(max(v, 0) / max_gender)  # NO min_r clamp for legend
            ms = r_data * ms_factor
            return max(ms, ms_min)  # tiny bins still visible but distinct

        legend_ms = [legend_markersize(v) for v in legend_mids]

        size_handles = [
            Line2D([], [], marker='o', linestyle='None', color='none',
                   markerfacecolor="#777777", markeredgecolor='black',
                   markersize=ms, alpha=0.9, label=lab)
            for ms, lab in zip(legend_ms, size_labels)
        ]

        gender_handles = [
            Line2D([], [], marker='o', linestyle='None', color='none',
                   markerfacecolor="green", markeredgecolor='black',
                   markersize=10, label="Female"),
            Line2D([], [], marker='o', linestyle='None', color='none',
                   markerfacecolor="orange", markeredgecolor='black',
                   markersize=10, label="Male"),
        ]

        # Anchor extra low to prevent overlapping content
        legend = ax.legend(
            handles=[Line2D([], [], linestyle="None", label="DISTANCE BANDS"),
                     *band_handles,
                     Line2D([], [], linestyle="None", label=""),
                     Line2D([], [], linestyle="None",
                            label="MIGRANT COUNT"), *size_handles,
                     Line2D([], [], linestyle="None", label=""),
                     Line2D([], [], linestyle="None", label="GENDER"), *gender_handles],
            loc='lower left', bbox_to_anchor=(0.012, 0.002),
            fontsize=10.5, frameon=True, edgecolor="#bdbdbd", facecolor="white",
            handletextpad=1.2, borderpad=0.9
        )
        ax.add_artist(legend)

        # border frame
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               linewidth=2.0, edgecolor='black',
                               facecolor='none', zorder=15))

        ax.axis("off")
        plt.tight_layout()

        out_png = os.path.join(output_path, filename)
        out_pdf = out_png.rsplit(".", 1)[0] + ".pdf"
        plt.savefig(out_png, dpi=400)
        plt.savefig(out_pdf)
        plt.close()

    def plot_stacked_percent_bars_by_county(
            self,
            data_loader,
            gender: str = "female",
            filename: str = "county_stacked_selected_vars.pdf",
    ):
        """
        Single-method, themed & tidy stacked plot (gender-aware).
        """
        gender = (gender or "female").strip().lower()

        # ---- helpers ----
        def to_share_pct(d):
            total = float(sum(d.values()))
            return {k: (0.0 if total == 0 else (float(v) / total) *
                                               100.0) for k, v in d.items()}

        def pct_get(d, k):
            try:
                return float(d.get(k, 0.0))
            except Exception:
                return 0.0

        def gini_pct(val):
            v = float(val)
            return v * 100.0 if v <= 1.0 else v

        # ---- data ----
        pop = data_loader.population
        gdp = data_loader.gdp
        dens = data_loader.population_density
        land = data_loader.land_area_km2
        hh_size = data_loader.avg_household_size
        working = getattr(data_loader, "working_population", {})

        reason_emp = data_loader.reason_employment
        reason_fd = data_loader.reason_forced_displacement
        reason_marriage = getattr(data_loader, "reason_marriage_formation", {})
        reason_family = getattr(data_loader, "reason_family_reunification", {})
        media_any = getattr(data_loader, "media_access_any_pct", {})
        media_none = getattr(data_loader, "media_access_none_pct", {})
        edu_none = getattr(data_loader, "edu_none", {})
        edu_some_primary = getattr(data_loader, "edu_some_primary", {})
        edu_some_secondary = getattr(data_loader, "edu_some_secondary", {})
        edu_completed_sec = getattr(data_loader, "edu_completed_secondary", {})
        emp_not_now = getattr(data_loader, "employed_not_currently", {})
        employed_none_12m = getattr(data_loader, "employed_none_last12m", {})
        food_acc = getattr(data_loader, "food_acceptable_pct", {})
        food_borderline = getattr(data_loader, "food_borderline_pct", {})
        gini = getattr(data_loader, "gini_coefficient", {})

        # ---- choose series + palettes ----
        if gender == "male":
            # Order controls stack order (top to bottom accumulates in this order).
            series_defs = [
                # Shares block
                ("Working pop (share)", to_share_pct(working), "share", "share"),
                ("GDP (share)", to_share_pct(gdp), "share", "share"),

                # Media / inequality
                ("No Media access (%)", media_none, "pct", "media"),
                ("Any Media access (%)", media_any, "pct", "media"),

                # Migration reasons (reds, together)
                ("Employment migration (%)", reason_emp, "pct", "reasons"),
                ("Marriage formation migration (%)", reason_marriage, "pct", "reasons"),
                ("Family reunification migration (%)", reason_family, "pct", "reasons"),
                ("Forced displacement migration (%)", reason_fd, "pct", "reasons"),

                # Education (blues, together)
                ("No Education (%)", edu_none, "pct", "edu"),
                ("Some primary Education (%)", edu_some_primary, "pct", "edu"),
                ("Some secondary Education (%)", edu_some_secondary, "pct", "edu"),
                ("Completed secondary Education (%)", edu_completed_sec, "pct", "edu"),

                # Jobs + Food
                ("Employed: none last 12 months (%)", employed_none_12m, "pct", "jobs"),
                ("Food: borderline (%)", food_borderline, "pct", "food"),
                ("Gini coefficient (%)", gini, "gini", "inequality"),
            ]
            palette = {
                "share": ["black", "#B45309"],  # cool grays
                "media": ["#66BB6A", "#2E7D32"],  # greens
                "reasons": ["#FF8A80", "#EF5350", "orange", "#B71C1C"],  # reds
                "edu": ["#0D47A1", "#1976D2", "#42A5F5", "#90CAF9"],  # blues
                "jobs": ["#6A1B9A"],  # royal purple
                "food": ["cyan"],  # teal
                "inequality": ["#8D6E63"],  # warm brown
            }
        else:
            series_defs = [
                # Shares block (five neat cool-grays)
                ("Population (share)", to_share_pct(pop), "share", "share"),
                ("GDP (share)", to_share_pct(gdp), "share", "share"),
                ("Density (share)", to_share_pct(dens), "share", "share"),
                ("Avg House hold size (share)", to_share_pct(hh_size), "share", "share"),

                # Reasons (reds)
                ("Employment migration (%)", reason_emp, "pct", "reasons"),
                ("Forced displacement migration (%)", reason_fd, "pct", "reasons"),

                # Jobs + Food
                ("Employed: not currently (%)", emp_not_now, "pct", "jobs"),
                ("Food: acceptable (%)", food_acc, "pct", "food"),
                ("Land area (share)", to_share_pct(land), "share", "share"),
            ]
            palette = {
                "share": ["#4B5563", "#B45309", "#9CA3AF", "limegreen", "#26A69A"],
                "reasons": ["#FF8A80", "#D32F2F"],
                "jobs": ["#6A1B9A"],
                "food": ["cyan"],
            }

        # ---- assign colors (group-cycling, stable) ----
        colors, used = [], defaultdict(int)
        for _, _, _, grp in series_defs:
            swatch = palette[grp]
            colors.append(swatch[used[grp] % len(swatch)])
            used[grp] += 1

        # ---- dataframe in region/county order ----
        rows = []
        for region, counties in self.region_order.items():
            for c in counties:
                r = {"County": c, "Region": region}
                for label, src, kind, grp in series_defs:
                    if kind == "share":
                        r[label] = float(src.get(c, 0.0))
                    elif kind == "pct":
                        r[label] = pct_get(src, c)
                    elif kind == "gini":
                        r[label] = gini_pct(src.get(c, 0.0))
                    else:
                        r[label] = 0.0
                rows.append(r)
        df = pd.DataFrame(rows)

        # ---- style ----
        mpl.rcParams.update({
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "font.size": 12,
        })

        categories = [lbl for (lbl, _, _, _) in series_defs]

        intra_spacing = 0.001
        inter_region_spacing = 0.002
        bar_width = intra_spacing * 0.7

        county_x_map, region_centers = {}, []
        x = 0.0
        for region, counties in self.region_order.items():
            if not counties:
                continue
            start_x = x
            for county in counties:
                county_x_map[county] = x
                x += intra_spacing
            end_x = x - intra_spacing
            region_centers.append(((start_x + end_x) / 2.0, region))
            x += inter_region_spacing

        df["x_plot"] = df["County"].map(county_x_map)
        df = df.sort_values("x_plot").reset_index(drop=True)

        # ---- plot ----
        fig, ax = plt.subplots(figsize=(16, 6))
        bottoms = np.zeros(len(df))

        for i, cat in enumerate(categories):
            ax.bar(
                df["x_plot"], df[cat].values,
                bottom=bottoms, color=colors[i],
                width=bar_width, edgecolor="white", linewidth=0.45, label=cat
            )
            bottoms += df[cat].values

        ax.set_xticks(df["x_plot"])
        ax.set_xticklabels(df["County"], rotation=90, fontsize=9)
        ax.set_ylabel("Percentage (%)", fontsize=15, fontweight="bold")
        ax.tick_params(axis="y", labelsize=12, length=4.5, width=1.1)
        ax.set_ylim(0, max(100, bottoms.max() * 1.12))

        # Region labels
        for center, region in region_centers:
            ax.text(center, -max(100, bottoms.max()) * 0.38, region,
                    ha="center", va="top", rotation=18,
                    fontsize=13, fontweight="bold", clip_on=False)

        handles, labels = ax.get_legend_handles_labels()
        plt.subplots_adjust(left=0.06, right=0.995, top=0.86, bottom=0.36)
        fig.legend(
            handles, labels,
            title="",
            fontsize=12, title_fontsize=14,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=4 if len(categories) > 10 else 3,
            frameon=False, handlelength=1.9,
        )

        os.makedirs(self.output_dir, exist_ok=True)
        outpath = os.path.join(self.output_dir, filename)
        plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.90])
        plt.savefig(outpath, dpi=400, bbox_inches="tight")
        plt.close()
        print(f"Stacked variables chart saved to {outpath}")

    def _normalize_county(self, name: str) -> str:
        name = (name or "").lower().replace("-", " ").replace("_", " ").strip()
        return {
            "homabay": "homa bay",
            "murang'a": "muranga",
            "taitataveta": "taita taveta",
            "tanariver": "tana river",
            "transnzoia": "trans nzoia",
            "uasingishu": "uasin gishu",
            "westpokot": "west pokot",
        }.get(name, name)

    def _make_county_number_map(self, counties: list[str]) -> dict[str, int]:
        return {c: i + 1 for i, c in enumerate(counties)}

    def _draw_top_number_name_legend(self, ax, county_number_map, color_lookup):
        ax.axis("off")
        sorted_pairs = sorted(county_number_map.items(), key=lambda x: x[1])  # by number
        n_cols, x_spacing, y_spacing, font_size = 6, 0.18, 0.16, 20
        for idx, (county, num) in enumerate(sorted_pairs):
            row, col = divmod(idx, n_cols)
            x = 0.02 + col * x_spacing
            y = 1.0 - row * y_spacing
            label = f"{num}: {county.title()}"
            ax.text(x, y, label,
                    transform=ax.transAxes,
                    fontsize=font_size,
                    color=color_lookup.get(str(num), "black"),
                    ha="left", va="top", fontweight="medium")

    def _render_dendrogram(self, ax, Z, county_number_map, cluster_threshold):
        numeric_labels = [str(county_number_map[c]) for c in county_number_map.keys()]
        dd = dendrogram(
            Z,
            labels=numeric_labels,
            leaf_rotation=90,
            leaf_font_size=14,
            color_threshold=cluster_threshold,
            ax=ax,
        )
        nice_map = {
            "C0": "#1f77b4", "C1": "#ff7f0e", "C2": "#2ca02c", "C3": "#d62728",
            "C4": "#9467bd", "C5": "#8c564b", "C6": "#e377c2", "C7": "#7f7f7f",
            "C8": "#bcbd22", "C9": "#17becf"
        }

        raw_leaf_colors = dict(zip(dd["ivl"], dd["leaves_color_list"]))
        self.dendo_colors = {lbl: nice_map.get(code, code) for lbl, code in
                             raw_leaf_colors.items()}

        for line in ax.get_lines():
            if max(line.get_ydata()) > 0:
                line.set_color("darkblue")
                line.set_linewidth(1.0)
        ax.set_ylabel("Cluster distance", fontsize=18, fontweight="bold", color="darkblue")
        ax.tick_params(axis="y", labelsize=12, width=1.2, length=8, colors="darkblue")
        ax.tick_params(axis="x", bottom=False)
        # color tick labels to match leaf colors
        for lbl in ax.get_xmajorticklabels():
            t = lbl.get_text()
            lbl.set_color(self.dendo_colors.get(t, "black"))
            lbl.set_fontweight("medium")
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)

    def _render_cluster_map_from_artifacts(self, ax, shapefile_gdf, counties,
                                           county_number_map, Z, cluster_threshold,
                                           name_col="NAME"):
        cluster_ids = fcluster(Z, t=cluster_threshold, criterion="distance")
        df = pd.DataFrame({
            "County": [self._normalize_county(c) for c in counties],
            "Number": [county_number_map[c] for c in counties],
            "Cluster": cluster_ids,
        })
        df["NumberStr"] = df["Number"].astype(int).astype(str)
        df["Color"] = df["NumberStr"].map(self.dendo_colors)

        gdf = shapefile_gdf.copy()
        if name_col not in gdf.columns and "County" in gdf.columns:
            name_col = "County"
        gdf = gdf.rename(columns={name_col: "County"})
        gdf["County"] = gdf["County"].apply(self._normalize_county)

        merged = gdf.merge(df, on="County", how="left")
        merged.plot(ax=ax, facecolor=merged["Color"].fillna("lightgrey"),
                    edgecolor="black", linewidth=0.5)

        for _, row in merged.iterrows():
            if row.geometry is not None and pd.notna(row.get("Number")):
                c = row.geometry.centroid
                t = ax.text(
                    c.x, c.y, str(int(row["Number"])),
                    fontsize=12, ha="center", va="center",
                    weight="bold", color="black", zorder=10,
                )
                t.set_path_effects([
                    patheffects.Stroke(linewidth=3.0, foreground="white"),
                    patheffects.Normal()
                ])
        ax.axis("off")
        for s in ax.spines.values():
            s.set_visible(False)

    def _shares_to_counts(self, share_s: pd.Series, pop_s: pd.Series) -> pd.Series:
        s = share_s.astype(str).str.replace(",", ".", regex=False)
        s = pd.to_numeric(s, errors="coerce")
        if s.max(skipna=True) > 1.5:  # looks like percent
            s = s / 100.0
        p = pd.to_numeric(pd.Series(pop_s), errors="coerce")
        return (s.reindex(p.index) * p).fillna(0.0)

    def _render_born_outside_vs_distance_from_artifacts(
            self, ax, y_series, distance_from_mandera,
            counties, county_number_map, mode: str = "percent"):
        def N(x):
            return self._normalize_county(x)

        s = y_series.copy()
        s = s.astype(str).str.replace(",", ".", regex=False)
        s = pd.to_numeric(s, errors="coerce")
        if mode == "percent" and s.max(skipna=True) <= 1.5:
            s = s * 100.0  # ensure percent scale

        s.index = [N(i) for i in s.index]

        d = {N(k): float(v) for k, v in distance_from_mandera.items()}
        d[N("Mandera")] = 0.0

        all_counties = sorted(set(counties) | set(s.index) | set(d.keys()))
        for c in all_counties:
            num = county_number_map.get(c)
            if num is None:
                continue
            label = str(num)
            color = self.dendo_colors.get(label, "gray")
            x, y = d.get(c), s.get(c)
            if pd.notna(x) and pd.notna(y):
                ax.scatter(x, y, color=color, edgecolor="black",
                           linewidth=0.35, s=55, zorder=3)
                ax.text(
                    x + 8, y, label, fontsize=12, ha="left", va="center",
                    weight="bold", color=color, zorder=5,
                    path_effects=[
                        patheffects.Stroke(linewidth=2.0, foreground="white"),
                        patheffects.Normal(),
                    ],
                )
        ax.set_xlabel("Distance from Mandera (km)", fontsize=16, fontweight="bold")
        ax.set_ylabel(
            "Born in Kenya but outside current county (%)"
            if mode == "percent"
            else "Born in Kenya but outside current county (count)",
            fontsize=16, fontweight="bold"
        )
        for side in ("top", "right"): ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(2.0)
            ax.spines[side].set_color("black")
            ax.spines[side].set_position(("outward", 8))
        ax.minorticks_off()
        ax.tick_params(axis="both", which="major",
                       direction="out", length=7, width=1.6, colors="black",
                       bottom=True, top=False, left=True, right=False, labelsize=12)

    def plot_combined_born_outside_view_for_gender(
            self, shapefile_gdf, data_loader, distance_from_mandera: dict[str, float],
            gender: str = "female", cluster_threshold: float = 1.5,
            save_path: str | None = None, shapefile_name_col: str = "NAME",
            mode: str = "percent"  # "percent" | "count"
    ):
        """
        Combined figure (single entry point):
          - top legend (number → name)
          - dendrogram (CountyClustering on chosen mode)
          - cluster map
          - scatter vs distance from Mandera
        """
        gender = (gender or "female").strip().lower()
        if save_path is None:
            suffix = "percent" if mode == "percent" else "count"
            save_path = os.path.join(self.output_dir, f"combined_born_outside_{gender}_{suffix}.pdf")

        # --- Pick series based on mode ---
        share_series = data_loader.data_df["Born_in_Kenya_but_outside"]
        if mode == "count":
            series_for_clustering = self._shares_to_counts(
                share_series, data_loader.data_df["Population"]
            )
        else:
            series_for_clustering = share_series

        # --- 1) CLUSTER via CountyClustering (with mode) ---
        cc = CountyClustering(immigrants=share_series, output_dir=self.output_dir,
                              gender=gender)
        _, labels, Z = cc.cluster_on_immigration_data(
            cluster_threshold=cluster_threshold,
            show_clusters=False,
            mode=mode,
            population=data_loader.data_df["Population"] if mode == "count" else None
        )
        counties = [self._normalize_county(c) for c in labels.index.tolist()]
        county_number_map = self._make_county_number_map(counties)

        # --- 2) LAYOUT ---
        fig = plt.figure(figsize=(25, 22), dpi=300)
        gs = gridspec.GridSpec(3, 6, height_ratios=[0.8, 2.5, 2.2],
                               width_ratios=[1.3, 1, 1, 1, 1, 1])
        gs.update(hspace=0.4, wspace=0.2)
        ax_abbrev = fig.add_subplot(gs[0, :])
        ax_dendro = fig.add_subplot(gs[1, 0:4])
        ax_map = fig.add_subplot(gs[1:3, 4:6])
        ax_scatter = fig.add_subplot(gs[2, 0:4])

        # enlarge map pane a touch
        box = ax_map.get_position()
        ax_map.set_position([box.x0 - 0.02, box.y0 - 0.09, box.width * 1.5,
                             box.height * 1.3])

        # --- 3) RENDERERS ---
        self._render_dendrogram(ax_dendro, Z, county_number_map, cluster_threshold)
        self._render_cluster_map_from_artifacts(
            ax=ax_map, shapefile_gdf=shapefile_gdf, counties=counties,
            county_number_map=county_number_map, Z=Z,
            cluster_threshold=cluster_threshold, name_col=shapefile_name_col
        )
        self._render_born_outside_vs_distance_from_artifacts(
            ax=ax_scatter, y_series=series_for_clustering,
            distance_from_mandera=distance_from_mandera,
            counties=counties, county_number_map=county_number_map, mode=mode
        )
        self._draw_top_number_name_legend(ax_abbrev, county_number_map,
                                          self.dendo_colors)

        # --- 4) SAVE ---
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=400, facecolor="white", transparent=False)
        plt.close(fig)
        print(f"Combined cluster view saved to: {save_path}")

    def _render_born_outside_vs_distance_from_artifacts2(
            self, ax, y_series, distance_from_mandera,
            counties, county_number_map):
        def N(x):
            return self._normalize_county(x)

        s = y_series.copy()
        s = s.astype(str).str.replace(",", ".", regex=False)
        s = pd.to_numeric(s, errors="coerce")
        s.index = [N(i) for i in s.index]

        d = {N(k): float(v) for k, v in distance_from_mandera.items()}
        d[N("Mandera")] = 0.0

        all_counties = sorted(set(counties) | set(s.index) | set(d.keys()))
        for c in all_counties:
            num = county_number_map.get(c)
            if num is None:
                continue
            label = str(num)
            color = self.dendo_colors.get(label, "gray")
            x, y = d.get(c), s.get(c)
            if pd.notna(x) and pd.notna(y):
                ax.scatter(x, y, color=color, edgecolor="black",
                           linewidth=0.35, s=55, zorder=3)
                ax.text(
                    x + 8, y, label, fontsize=12, ha="left", va="center",
                    weight="bold", color=color, zorder=5,
                    path_effects=[
                        patheffects.Stroke(linewidth=2.0, foreground="white"),
                        patheffects.Normal(),
                    ],
                )
        ax.set_xlabel("Distance from Mandera (km)", fontsize=16, fontweight="bold")
        ax.set_ylabel("Born in Kenya but outside current county (%)",
                      fontsize=16, fontweight="bold")
        for side in ("top", "right"): ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(2.0)
            ax.spines[side].set_color("black")
            ax.spines[side].set_position(("outward", 8))
        ax.minorticks_off()
        ax.tick_params(axis="both", which="major",
                       direction="out", length=7, width=1.6, colors="black",
                       bottom=True, top=False, left=True, right=False, labelsize=12)

    def plot_combined_born_outside_view_for_gender2(
            self, shapefile_gdf, data_loader, distance_from_mandera: dict[str, float],
            gender: str = "female", cluster_threshold: float = 1.5,
            save_path: str | None = None, shapefile_name_col: str = "NAME",
    ):
        """
        DRY combined figure:
          - top legend (number → name)
          - dendrogram (uses CountyClustering once)
          - cluster map (reuses dendrogram colors)
          - scatter vs distance from Mandera (reuses colors)
        """

        gender = (gender or "female").strip().lower()
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"combined_born_outside_{gender}.pdf")

        # --- 1) CLUSTER ONCE via CountyClustering (no scaling here in Plotter) ---
        series = data_loader.data_df["Born_in_Kenya_but_outside"]
        cc = CountyClustering(immigrants=series, output_dir=self.output_dir,
                              gender=gender)
        _, labels, Z = cc.cluster_on_immigration_data(
            cluster_threshold=cluster_threshold, show_clusters=False)
        counties = [self._normalize_county(c) for c in labels.index.tolist()]
        county_number_map = self._make_county_number_map(counties)

        # --- 2) LAYOUT ---
        fig = plt.figure(figsize=(25, 22), dpi=300)
        gs = gridspec.GridSpec(3, 6, height_ratios=[0.8, 2.5, 2.2],
                               width_ratios=[1.3, 1, 1, 1, 1, 1])
        gs.update(hspace=0.4, wspace=0.2)
        ax_abbrev = fig.add_subplot(gs[0, :])
        ax_dendro = fig.add_subplot(gs[1, 0:4])
        ax_map = fig.add_subplot(gs[1:3, 4:6])
        ax_scatter = fig.add_subplot(gs[2, 0:4])

        # enlarge map pane a touch
        box = ax_map.get_position()
        ax_map.set_position([box.x0 - 0.02, box.y0 - 0.09, box.width * 1.5,
                             box.height * 1.3])

        # --- 3) RENDERERS (no recomputation) ---
        self._render_dendrogram(ax_dendro, Z, county_number_map,
                                cluster_threshold)
        self._render_cluster_map_from_artifacts(
            ax=ax_map, shapefile_gdf=shapefile_gdf, counties=counties,
            county_number_map=county_number_map, Z=Z,
            cluster_threshold=cluster_threshold, name_col=shapefile_name_col
        )
        self._render_born_outside_vs_distance_from_artifacts(
            ax=ax_scatter, y_series=series, distance_from_mandera=distance_from_mandera,
            counties=counties, county_number_map=county_number_map
        )
        self._draw_top_number_name_legend(ax_abbrev, county_number_map,
                                          self.dendo_colors)

        # --- 4) SAVE ---
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=400, facecolor="white", transparent=False)
        plt.close(fig)
        print(f"Combined cluster view saved to: {save_path}")








