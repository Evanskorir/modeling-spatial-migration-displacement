"""Clean figures for the nine-variable migration redesign.

Use with the included runner.py and descriptives.py update. Requires only the existing NumPy,
pandas and Matplotlib dependencies. Every rendering operation is contained here.
The correlation layout adapts the migration reference; county map conventions
adapt the supplied GeographicMapPlotter with a shared five-class percentage bar.
Plot styling never transforms outcomes, recomputes model estimates or selects
predictors. Uncertainty panels display the supplied model-based intervals.
"""
from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Polygon as MplPolygon, Rectangle
import matplotlib.patheffects as path_effects
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from .data_loader import county_key
from .variables import CANDIDATES, FEATURE_LABELS


def _themed(method):
    """Scope style changes to this figure; do not change the caller's rcParams."""
    @wraps(method)
    def draw(self, *args, **kwargs):
        with plt.rc_context(self.STYLE):
            return method(self, *args, **kwargs)
    return draw


class Plotter:
    """One output directory, one consistent theme, existing public interfaces.

    Existing calls remain valid: Plotter(folder), plot_county_comparison(paired),
    candidate_distributions(raw), plot_maps(paired, geojson_path),
    correlation(corr, name), vif(table, threshold, name), and
    model_results(effects, predictions). Each saves a vector PDF and a PNG.

    Optional correlation_difference(female_corr, male_corr) compares predictor
    correlations descriptively; it is not a test of gender differences.
    """
    BG = "#FFFFFF"
    INK = "#19343E"
    MUTED = "#60747A"
    GRID = "#E2E7E7"
    FEMALE = "#C65D49"
    MALE = "#267D88"
    STYLE = {
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.labelsize": 10, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#CAD3D5", "axes.linewidth": .7,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": BG, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "axes.unicode_minus": True,
    }
    SHORT_LABELS = {
        "log_population_density": "Log population density",
        "poverty_rate": "Poverty rate",
        "currently_employed": "Current employment",
        "completed_secondary": "Completed secondary",
        "household_size": "Household size",
        "mass_media_access": "Mass media access",
        "daily_internet_access": "Daily internet access",
        "gini_coefficient": "Gini coefficient",
        "acceptable_food_consumption": "Acceptable food consumption",
    }

    def __init__(self, output_dir, dpi=350, formats=("pdf", "png")):
        self.figures_dir = Path(output_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = int(dpi)
        if self.dpi < 72:
            raise ValueError("dpi must be at least 72")
        self.formats = tuple(dict.fromkeys(formats))
        if not self.formats or set(self.formats)-{"pdf", "png", "svg"}:
            raise ValueError("formats must contain pdf, png and/or svg")
        self.outputs = []

    def _label(self, key, width=None):
        value = self.SHORT_LABELS.get(key, FEATURE_LABELS.get(key, str(key).replace("_", " ")))
        return textwrap.fill(value, width) if width else value

    def _sex(self):
        parts = [p.casefold() for p in self.figures_dir.parts]
        return "Women" if "female" in parts else "Men" if "male" in parts else ""

    def _color(self):
        return self.FEMALE if self._sex()=="Women" else self.MALE

    def _heading(self, fig, title, subtitle="", left=.06):
        """Compatibility hook: figure titles/subtitles intentionally suppressed."""
        return None

    def _footnote(self, fig, text, left=.06):
        """Explanatory notes belong in the caption/README, not on the figure."""
        return None

    def _style_axes(self, ax, grid="x"):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#C5D0D2")
        ax.tick_params(length=0, pad=7)
        ax.set_axisbelow(True)
        if grid:
            ax.grid(axis=grid, color=self.GRID, linewidth=.65)

    def _save_figure(self, fig, name):
        """Close in finally even when an export fails; keep outputs unique."""
        name = Path(name)
        if name.name != str(name) or name.suffix in {".pdf", ".png", ".svg"}:
            raise ValueError("Figure name must be a plain filename stem")
        try:
            for suffix in self.formats:
                path = self.figures_dir / f"{name}.{suffix}"
                fig.savefig(path, dpi=self.dpi, bbox_inches="tight", pad_inches=.18,
                            facecolor=self.BG)
                if path not in self.outputs:
                    self.outputs.append(path)
        finally:
            plt.close(fig)

    @staticmethod
    def _require(frame, columns):
        missing = set(columns)-set(frame.columns)
        if missing:
            raise ValueError(f"Missing plot columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("Cannot plot an empty table")

    @staticmethod
    def _finite(frame, columns):
        if not np.isfinite(frame[list(columns)].to_numpy(float)).all():
            raise ValueError("Nonfinite figure input; inspect source values")

    @staticmethod
    def _validate_paired(paired):
        Plotter._require(paired, ["County", "county_key", "female_pct", "male_pct", "gap_pp"])
        Plotter._finite(paired, ["female_pct", "male_pct", "gap_pp"])
        if paired.county_key.duplicated().any():
            raise ValueError("Duplicate county in paired figure input")
        if not paired[["female_pct", "male_pct"]].apply(lambda s: s.between(0,100)).all().all():
            raise ValueError("Prevalence figure inputs must use the 0–100 scale")
        if not np.allclose(paired.gap_pp, paired.female_pct-paired.male_pct, atol=1e-8):
            raise ValueError("Gender gap must be female minus male in percentage points")

    @_themed
    def plot_county_comparison(self, paired):
        self._validate_paired(paired)
        d = paired.sort_values(["gap_pp", "County"], ascending=[False, True]).reset_index(drop=True)
        n = len(d); y = np.arange(n)
        fig = plt.figure(figsize=(12, max(10, .245*n+2.2)))
        gs = fig.add_gridspec(1, 2, left=.18, right=.95, top=.82, bottom=.085,
                              width_ratios=[1.55, 1], wspace=.14)
        ax = fig.add_subplot(gs[0]); gap = fig.add_subplot(gs[1], sharey=ax)
        self._heading(fig, "County differences in migration prevalence",
                      f"{n} paired county estimates · Ranked by the female–male gap", left=.055)
        stats = [("WOMEN · MEAN COUNTY PREVALENCE", f"{d.female_pct.mean():.1f}%", self.FEMALE),
                 ("MEN · MEAN COUNTY PREVALENCE", f"{d.male_pct.mean():.1f}%", self.MALE),
                 ("MEAN COUNTY GAP", f"{d.gap_pp.mean():+.1f} pp", self.INK)]
        for x, (label, value, color) in zip([.18, .46, .76], stats):
            fig.text(x, .871, label, fontsize=7.5, weight="semibold", color=self.MUTED)
            fig.text(x, .839, value, fontsize=20, weight="bold", color=color)
        for a in [ax, gap]:
            for i in range(0, n, 2):
                a.axhspan(i-.5, i+.5, color="#F0F1ED", zorder=0)
            self._style_axes(a)
        ax.hlines(y, d.male_pct, d.female_pct, color="#B9C5C7", linewidth=1.15)
        ax.scatter(d.male_pct, y, s=27, color=self.MALE, edgecolors=self.BG, linewidths=.45, zorder=3)
        ax.scatter(d.female_pct, y, s=27, color=self.FEMALE, edgecolors=self.BG, linewidths=.45, zorder=4)
        ax.set_yticks(y, d.County, fontsize=8.5); ax.set_ylim(n-.5, -.5)
        ax.set_xlim(-2, 102); ax.set_xticks([0,25,50,75,100])
        ax.set_xlabel("Recorded prevalence (%)", labelpad=10)
        handles = [Line2D([],[],marker="o",linestyle="",color=c,label=s,markersize=6)
                   for s,c in [("Women",self.FEMALE),("Men",self.MALE)]]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.18,.906),
                   ncol=2, frameon=False, fontsize=9, handletextpad=.35, borderaxespad=0)
        colors = np.where(d.gap_pp>=0, self.FEMALE, self.MALE)
        gap.hlines(y, 0, d.gap_pp, colors=colors, linewidth=2, alpha=.75)
        gap.scatter(d.gap_pp,y,c=colors,s=19,zorder=3)
        gap.axvline(0,color="#9CAEB1",linewidth=.9)
        limit = max(10., d.gap_pp.abs().max()+13)
        gap.set_xlim(-limit,limit); gap.tick_params(labelleft=False)
        gap.xaxis.set_major_locator(MaxNLocator(5))
        gap.set_xlabel("Women minus men (pp)", labelpad=10)
        for row, value in enumerate(d.gap_pp):
            gap.text(value+(2 if value>=0 else -2),row,f"{value:+.1f}",
                     ha="left" if value>=0 else "right",va="center",fontsize=7.5,color=self.MUTED)
        self._footnote(fig, "County averages are unweighted, not national prevalence estimates.\n"
                       "Gaps are descriptive; no county-level significance tests are implied.", left=.18)
        self._save_figure(fig, "county_gender_comparison")

    @_themed
    def candidate_distributions(self, raw):
        self._require(raw, ["sex", "county_key"]+[c[0] for c in CANDIDATES])
        if set(raw.sex.unique()) != {"female", "male"}:
            raise ValueError("Candidate distributions require female and male rows")
        if raw.duplicated(["sex","county_key"]).any():
            raise ValueError("Duplicate county–sex distribution input")
        self._finite(raw,[c[0] for c in CANDIDATES])
        fig, axes = plt.subplots(3,3,figsize=(13.3,11.7))
        fig.subplots_adjust(left=.07,right=.98,bottom=.05,top=.97,hspace=.35,wspace=.28)
        self._heading(fig,"Nine domains of county context",
                      "County distributions by sex · Individual points, interquartile range and median",left=.07)
        for index,(ax,(source,key,method,domain)) in enumerate(zip(axes.flat,CANDIDATES)):
            self._style_axes(ax,grid="y")
            arrays=[]
            subsets=[]
            for sex in ["female","male"]:
                subset=raw.loc[raw.sex.eq(sex)].sort_values('county_key')
                subsets.append(subset.set_index('county_key')[source])
                arrays.append(subset[source].to_numpy(float))
            # Density is modelled on a log scale; violin geometry is also computed
            # on log10 density here, and its axis is explicitly labelled as such.
            if method=="natural_log":
                if any((a<=0).any() for a in arrays): raise ValueError("Density must be positive")
                arrays=[np.log10(a) for a in arrays]
            for xpos, values, color in zip([0,1],arrays,[self.FEMALE,self.MALE]):
                if len(values)>1 and np.std(values)>1e-12:
                    body=ax.violinplot([values],positions=[xpos],widths=.64,
                                       showmeans=False,showextrema=False,showmedians=False)['bodies'][0]
                    path=body.get_paths()[0]
                    path.vertices[:,0]=np.minimum(path.vertices[:,0],xpos)
                    body.set(facecolor=color,edgecolor='none',alpha=.22)
                # Deterministic point spreading; no random variation between runs.
                jitter=.10+.16*((np.arange(len(values))*.61803398875)%1)
                ax.scatter(xpos+jitter,values,s=13,color=color,alpha=.65,linewidths=.25,
                           edgecolors=self.BG,zorder=3)
                q1,median,q3=np.quantile(values,[.25,.5,.75])
                ax.plot([xpos,xpos],[q1,q3],color=color,linewidth=4,solid_capstyle='round',zorder=4)
                ax.scatter([xpos],[median],s=30,facecolor=self.BG,edgecolors=color,linewidths=1.3,zorder=5)
            ax.set_xlim(-.45,1.45); ax.set_xticks([0,1],["Women","Men"])
            shared = (set(subsets[0].index)==set(subsets[1].index) and
                      np.allclose(subsets[0].sort_index(),subsets[1].sort_index()))
            title=self._label(key).replace("Log population density","Population density")
            ax.set_title(title,loc='left',pad=10,fontsize=10.5)
            if method=="natural_log":
                lo=min(a.min() for a in arrays);hi=max(a.max() for a in arrays)
                ticks=np.arange(np.floor(lo),np.ceil(hi)+1)
                ax.set_yticks(ticks,[f"$10^{{{int(v)}}}$" for v in ticks]);unit="Source density · log scale"
            elif method=="percent_divided_by_100":unit="Percent"
            else:unit="People per household" if key=="household_size" else "Gini · source scale"
            ax.set_ylabel(unit,fontsize=9)
            ax.margins(y=.15)
        self._footnote(fig,"Each point is a county estimate. Shared measures repeat the same county values for both sexes.\n"
                       "Density uses a logarithmic display; the remaining panels use source units. Education source-quality flags remain.",left=.07)
        self._save_figure(fig,"nine_candidate_distributions")

    @staticmethod
    def _check_correlation(corr, symmetric=True):
        if corr.empty or corr.index.duplicated().any() or corr.columns.duplicated().any():
            raise ValueError("Correlation labels must be nonempty and unique")
        if list(corr.index)!=list(corr.columns):
            raise ValueError("Correlation rows and columns must have identical order")
        a=corr.to_numpy(float)
        if not np.isfinite(a).all():raise ValueError("Nonfinite correlation input")
        if symmetric and (not np.allclose(a,a.T,atol=1e-8) or
                          not np.allclose(np.diag(a),1,atol=1e-8) or np.max(abs(a))>1+1e-8):
            raise ValueError("Expected a symmetric correlation matrix with diagonal 1")
        return a

    def _correlation_panel(self, corr, name, title="", subtitle="", difference=False):
        a = self._check_correlation(corr, symmetric=not difference)
        n = len(corr)
        limit = 2 if difference else 1
        fig, ax = plt.subplots(figsize=(max(8.4,.72*n+2.2), max(8,.66*n+2.1)))
        fig.subplots_adjust(left=.24,right=.98,bottom=.26,top=.98)
        ax.imshow(a[::-1],cmap='coolwarm',norm=Normalize(-limit,limit),interpolation='nearest')
        labels = [self._label(c) for c in corr.columns]
        ax.set_xticks(range(n),labels,rotation=90,ha='center',fontsize=9)
        ax.set_yticks(range(n),labels[::-1],fontsize=9)
        ax.tick_params(length=0,pad=8)
        ax.set_xticks(np.arange(-.5,n,1),minor=True)
        ax.set_yticks(np.arange(-.5,n,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1.2)
        ax.tick_params(which='minor',length=0)
        for i in range(n):
            for j in range(n):
                value = a[n-1-i,j]
                ax.text(j,i,f"{value:+.2f}" if difference else f"{value:.2f}",
                        ha='center',va='center',fontsize=9,
                        color='white' if abs(value)>limit*.60 else self.INK,
                        weight='semibold')
        for spine in ax.spines.values():
            spine.set(color='#899A9E',linewidth=.8)
        self._save_figure(fig,name)

    @_themed
    def correlation(self,corr,name):
        stage="After correlation screening" if 'retained' in str(name) else "Initial candidate set"
        sex=self._sex()
        self._correlation_panel(corr,name,f"Predictor relationships{' · '+sex if sex else ''}",
                                f"{stage} · {len(corr)} predictors · Consistent −1 to +1 colour scale")

    @_themed
    def correlation_difference(self,female_corr,male_corr,name='gender_correlation_difference'):
        self._check_correlation(female_corr);self._check_correlation(male_corr)
        if set(female_corr.columns)!=set(male_corr.columns):
            raise ValueError("Compare the same predictor set in both sexes")
        aligned=male_corr.loc[female_corr.index,female_corr.columns]
        self._correlation_panel(aligned-female_corr,name,"Differences in predictor relationships",
                                "Men minus women · Identical predictor definitions and ordering",difference=True)

    @staticmethod
    def _polygon_path(rings):
        """Opposite ring orientations preserve polygon holes with nonzero fill."""
        vertices,codes=[],[]
        for index,ring in enumerate(rings):
            xy=np.asarray(ring,dtype=float)
            if xy.ndim!=2 or xy.shape[0]<4 or xy.shape[1]<2:
                raise ValueError("Invalid GeoJSON polygon ring")
            xy=xy[:,:2]
            if not np.isfinite(xy).all() or (abs(xy[:,0])>180).any() or (abs(xy[:,1])>90).any():
                raise ValueError("Map input must contain finite WGS84 longitude/latitude")
            if not np.allclose(xy[0],xy[-1]):xy=np.vstack([xy,xy[0]])
            area=np.sum(xy[:-1,0]*xy[1:,1]-xy[1:,0]*xy[:-1,1])
            if (index==0 and area<0) or (index>0 and area>0):xy=xy[::-1]
            vertices.extend(xy);codes.extend([MplPath.MOVETO]+[MplPath.LINETO]*(len(xy)-2)+[MplPath.CLOSEPOLY])
        return MplPath(vertices,codes)

    def _read_geometry(self,paired,geojson_path):
        path=Path(geojson_path) if geojson_path else None
        if path is None or not path.is_file():raise FileNotFoundError(f"County GeoJSON not found: {path}")
        data=json.loads(path.read_text());crs=data.get('crs',{}).get('properties',{}).get('name','')
        if crs and not any(s in crs.upper() for s in ['4326','CRS84']):
            raise ValueError("County GeoJSON must use WGS84 longitude/latitude")
        geometry,joins={},[]
        for feature in data['features']:
            props=feature['properties']
            name=next((props.get(c) for c in ['NAME_1','NAME','County','county'] if props.get(c)),None)
            if name is None:raise ValueError("No supported county-name property in GeoJSON")
            key=county_key(name)
            if key in geometry:raise ValueError(f"Duplicate county geometry: {name}")
            geom=feature['geometry']
            if geom['type'] not in ['Polygon','MultiPolygon']:raise ValueError(f"Unsupported county geometry: {name}")
            rings=[geom['coordinates']] if geom['type']=='Polygon' else geom['coordinates']
            geometry[key]=[self._polygon_path(p) for p in rings]
            joins.append(dict(county_key=key,geojson_county=name))
        expected=set(paired.county_key)
        if set(geometry)!=expected:
            raise ValueError(f"County map mismatch: missing={sorted(expected-set(geometry))}; extra={sorted(set(geometry)-expected)}")
        return geometry,paired[['County','county_key']].merge(pd.DataFrame(joins),on='county_key',validate='one_to_one')

    @staticmethod
    def _interior_anchor(paths):
        """Find a deterministic interior display anchor; no geographic inference."""
        path=max(paths,key=lambda p:np.prod(np.ptp(p.vertices,axis=0)))
        low=path.vertices.min(axis=0);high=path.vertices.max(axis=0)
        center=(low+high)/2
        if path.contains_point(center):return center
        xx,yy=np.meshgrid(np.linspace(low[0],high[0],25),np.linspace(low[1],high[1],25))
        points=np.column_stack([xx.ravel(),yy.ravel()]);inside=points[path.contains_points(points)]
        if len(inside):return inside[np.argmin(np.sum((inside-center)**2,axis=1))]
        return path.vertices[0]

    MAP_COLORS = ("#ffe9a0", "#ffc06a", "#ff9249", "#ff633d", "#f51c3a")
    SMALL_COUNTIES = {
        "mombasa", "busia", "vihiga", "kisii", "nyamira", "bomet", "kericho",
        "nandi", "kiambu", "nairobi", "kirinyaga", "muranga", "embu",
        "transnzoia", "uasingishu", "elgeyomarakwet", "nyandarua", "nyeri",
    }

    @staticmethod
    def _cartographic_imports():
        try:
            from shapely.geometry import shape, Point, Polygon
            from shapely.ops import unary_union, nearest_points
            from pyproj import Geod
        except ImportError as exc:
            raise ImportError('Labelled maps require shapely>=2 and pyproj>=3. '
                              'Install requirements_plotting.txt or use --no-maps.') from exc
        return shape, Point, Polygon, unary_union, nearest_points, Geod

    @staticmethod
    def _draw_outline(ax, geom, **kwargs):
        polygons = [geom] if geom.geom_type=='Polygon' else list(geom.geoms)
        for poly in polygons:
            if poly.is_empty or poly.geom_type!='Polygon':
                continue
            xy=np.asarray(poly.exterior.coords)
            ax.plot(xy[:,0],xy[:,1],**kwargs)

    @staticmethod
    def _compass(ax, x=0.94, y=0.85, size=0.025):
        """
        Compact compass positioned at middle-right,
        similar to the IEBC national map.
        """
        transform = ax.transAxes
        ax.text(
            x,
            y + size * 1.55,
            "N",
            transform=transform,
            ha="center",
            va="center",
            fontsize=8,
            weight="bold",
            zorder=150,
        )
        ax.plot(
            [x, x],
            [y - size * 0.9, y + size * 0.9],
            transform=transform,
            color="black",
            linewidth=0.8,
            zorder=140,
        )
        ax.plot(
            [x - size * 0.9, x + size * 0.9],
            [y, y],
            transform=transform,
            color="black",
            linewidth=0.8,
            zorder=140,
        )
        north = MplPolygon(
            [
                (x, y + size * 1.25),
                (x - size * 0.2, y + size * 0.04),
                (x + size * 0.2, y + size * 0.04),
            ],
            transform=transform,
            closed=True,
            facecolor="black",
            edgecolor="black",
            linewidth=0.6,
            zorder=150,
        )
        ax.add_patch(north)
        south = MplPolygon(
            [
                (x, y - size * 1.05),
                (x - size * 0.16, y - size * 0.02),
                (x + size * 0.16, y - size * 0.02),
            ],
            transform=transform,
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
            zorder=150,
        )
        ax.add_patch(south)
        east = MplPolygon(
            [
                (x + size * 0.95, y),
                (x + size * 0.03, y + size * 0.13),
                (x + size * 0.03, y - size * 0.13),
            ],
            transform=transform,
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
            zorder=150,
        )
        ax.add_patch(east)
        west = MplPolygon(
            [
                (x - size * 0.95, y),
                (x - size * 0.03, y + size * 0.13),
                (x - size * 0.03, y - size * 0.13),
            ],
            transform=transform,
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
            zorder=150,
        )
        ax.add_patch(west)

    def _covariate_county_labels(self, ax, rows, values, small_lookup, Point, nearest_points):
        """Covariate reference: named interiors, western callouts, indexed small units."""
        callouts={'transnzoia','bungoma','busia','siaya','homabay','migori','mombasa'}
        import re
        normalize=lambda name: re.sub('[^a-z]','',name.casefold())
        bounds=np.array([r['geometry'].bounds for r in rows])
        width=bounds[:,2].max()-bounds[:,0].min();height=bounds[:,3].max()-bounds[:,1].min()
        western=sorted([r for r in rows if normalize(r['County']) in callouts-{'mombasa'}],
                       key=lambda r:-r['geometry'].representative_point().y)
        west_y={};previous=None
        for row in western:
            y=row['geometry'].representative_point().y
            if previous is not None:y=min(y,previous-.047*height)
            west_y[row['key']]=y;previous=y
        for row in rows:
            geom=row['geometry'];point=geom.representative_point();key=normalize(row['County'])
            name=row['County'];value=f"{values[row['key']]:.0f}%"
            if key in callouts:
                if key in {'transnzoia','bungoma'}:
                    shift=.08 if key=='transnzoia' else -.04
                    target=Point(point.x,point.y+shift*(geom.bounds[3]-geom.bounds[1]))
                    if not geom.covers(target):target=point
                    x=geom.bounds[0]-.03*width;y=target.y;end=target
                else:
                    x,y=((geom.bounds[2]+.018*width,point.y-.015*height) if key=='mombasa'
                         else (geom.bounds[0]-.009*width,west_y[row['key']]))
                    edge=nearest_points(geom.boundary,Point(x,y))[0]
                    end=Point(edge.x+.68*(point.x-edge.x),edge.y+.68*(point.y-edge.y))
                    if not geom.covers(end):end=point
                east=key=='mombasa';offset=.006*width if east else -.006*width
                ax.plot([x+(-.004 if east else .004)*width,end.x],[y+.008*height,end.y],
                        color='#303030',lw=1.5,zorder=65,clip_on=False,solid_capstyle='round')
                ax.text(x+offset,y+.005*height,name,ha='left' if east else 'right',va='bottom',
                        fontsize=8,weight='bold',color='black',zorder=66,clip_on=False)
                ax.text(x+offset,y+.003*height,value,ha='left' if east else 'right',va='top',
                        fontsize=7.8,color='black',zorder=66,clip_on=False)
            elif row['key'] in small_lookup:
                ax.text(point.x,point.y,str(small_lookup[row['key']]),ha='center',va='center',
                        fontsize=7.5,weight='bold',color='black',zorder=60)
            else:
                offsets={'nakuru':(-.22,-.04),'nyandarua':(-.03,.16),
                         'nyeri':(.40,.15),'meru':(.04,.08)}
                dx,dy=offsets.get(key,(0,0));point=Point(point.x+dx,point.y+dy)
                label=ax.annotate(name,(point.x,point.y),xytext=(0,2),textcoords='offset points',
                                  ha='center',va='bottom',fontsize=8,weight='bold',color='black',zorder=60)
                label.set_path_effects([path_effects.withStroke(linewidth=.7,foreground='white')])
                ax.annotate(value,(point.x,point.y),xytext=(0,-2),textcoords='offset points',
                            ha='center',va='top',fontsize=7.8,color='black',zorder=60)

    @_themed
    def plot_maps(self,paired,geojson_path):
        """Covariate-style names, western callouts and a shared small-county key.

        No female-minus-male map. One common vertical colour bar. Maps adapt the
        supplied cholera palette, ocean wedge, outlines, compass and callouts.
        """
        self._validate_paired(paired)
        geometry,join=self._read_geometry(paired,geojson_path)
        shape,Point,Polygon,unary_union,nearest_points,Geod=self._cartographic_imports()
        raw=json.loads(Path(geojson_path).read_text())
        shapes={}
        for feature in raw['features']:
            props=feature['properties']
            name=next(props[c] for c in ['NAME_1','NAME','County','county'] if props.get(c))
            geom=shape(feature['geometry'])
            if not geom.is_valid or geom.is_empty:
                raise ValueError(f'Invalid county polygon: {name}; do not repair silently')
            shapes[county_key(name)]=geom
        outline=unary_union(list(shapes.values()))
        bounds=outline.bounds;minx,miny,maxx,maxy=bounds
        width,height=maxx-minx,maxy-miny
        key=paired[['County','county_key']].sort_values('County',key=lambda v:v.str.casefold()).reset_index(drop=True)
        # Use the reference's equal-area 38th-percentile rule for indexed counties.
        from pyproj import Transformer
        from shapely.ops import transform
        project=Transformer.from_crs('EPSG:4326','EPSG:6933',always_xy=True).transform
        areas={k:transform(project,g).area for k,g in shapes.items()}
        cutoff=float(np.quantile(list(areas.values()),.38))
        callout_keys={county_key(n) for n in ['Trans Nzoia','Bungoma','Busia','Siaya','Homa Bay','Migori','Mombasa']}
        small_keys=[k for k in key.county_key if areas[k]<cutoff and k not in callout_keys]
        small_lookup={k:i+1 for i,k in enumerate(small_keys)}
        key['county_number']=key.county_key.map(small_lookup).astype('Int64')
        number_lookup=key.set_index('county_key').county_number.to_dict()
        join=join.merge(key[['county_key','county_number']],on='county_key',validate='one_to_one')
        rows=[dict(County=row.County,key=row.county_key,geometry=shapes[row.county_key],
                   number=number_lookup[row.county_key]) for row in paired.itertuples()]
        key.to_csv(self.figures_dir/'county_number_key.csv',index=False)
        ocean_base=Polygon([(40.45,-1.55),(41.5,-1.75),(41.2,-5.05),(39.3,-4.95),
                            (39.35,-4.45),(39.55,-3.95),(39.75,-3.2),(40.1,-2.35)])
        ocean=ocean_base.difference(outline)
        fig=plt.figure(figsize=(22,11))
        axes=[fig.add_axes([.008+i*.372,.10,.350,.84]) for i in range(2)]
        cmap=ListedColormap(self.MAP_COLORS)
        boundaries=np.linspace(0,100,6)
        norm=BoundaryNorm(boundaries,cmap.N,clip=True)
        for ax,col,title in zip(axes,['female_pct','male_pct'],['Women','Men']):
            ocean_polys=[ocean] if ocean.geom_type=='Polygon' else list(ocean.geoms)
            for poly in ocean_polys:
                if poly.geom_type=='Polygon':
                    ax.add_patch(PathPatch(self._polygon_path([poly.exterior.coords]+[r.coords for r in poly.interiors]),
                                           facecolor='#BFDDF4',edgecolor='#768591',lw=.35,zorder=0))
            for row in paired.itertuples():
                for path in geometry[row.county_key]:
                    ax.add_patch(PathPatch(path,facecolor=cmap(norm(getattr(row,col))),
                                           edgecolor='0.52',linewidth=.55,zorder=10))
            self._draw_outline(ax,outline,color='0.05',linewidth=1.05,zorder=40)
            ax.set_xlim(minx-.18*width,maxx+.10*width)
            ax.set_ylim(min(miny-.045*height,-5.15),maxy+.045*height)
            ax.set_aspect('equal',adjustable='box');ax.axis('off')
            ax.set_title(title,fontsize=13,weight='semibold',pad=9)
            ax.text(40.75,-3.55,'INDIAN\nOCEAN',fontsize=7,ha='center',va='center',
                    style='italic',color='#334155',zorder=60)
            self._compass(ax)
            self._covariate_county_labels(ax,rows,paired.set_index('county_key')[col].to_dict(),
                                           small_lookup,Point,nearest_points)
        # One vertical bar with exactly the same fixed percentage classes as
        # the cholera triptych. No period-specific quantile reclassification.
        cax=fig.add_axes([.80,.12,.012,.25])
        bar=fig.colorbar(ScalarMappable(cmap=cmap,norm=norm),cax=cax,
                        boundaries=boundaries,ticks=boundaries,spacing='uniform',
                        orientation='vertical',drawedges=True)
        bar.outline.set_edgecolor('#626262');bar.outline.set_linewidth(.55)
        bar.dividers.set_color('white');bar.dividers.set_linewidth(.8)
        bar.ax.set_yticklabels([f'{int(v)}%' for v in boundaries])
        bar.ax.tick_params(labelsize=10,length=0,pad=6,colors='#303030')
        cax.set_title('Migration (%)',fontsize=8.4,weight='semibold',pad=9)
        # One geodesically calibrated east-west distance bar at midpoint latitude.
        latitude=(miny+maxy)/2
        endpoint,_,_=Geod(ellps='WGS84').fwd(minx,latitude,90,200000)
        fig.canvas.draw()
        transform=fig.transFigure.inverted().transform
        a=transform(axes[0].transData.transform((minx,latitude)))
        b=transform(axes[0].transData.transform((endpoint,latitude)))
        length=b[0]-a[0];left=.055;y=.10
        for j,color in enumerate(['black','white']):
            fig.add_artist(Rectangle((left+j*length/2,y),length/2,.007,
                                     transform=fig.transFigure,facecolor=color,
                                     edgecolor='black',lw=.7,zorder=150))
        for fraction,label in [(0,'0'),(.5,'100'),(1,'200 km')]:
            fig.text(left+fraction*length,y+.013,label,ha='center',fontsize=7,color='black')
        # Only the indexed small/central counties appear in the shared key.
        fig.text(.765,.84,'Indexed counties',fontsize=11,weight='semibold',color=self.INK)
        fig.text(.930,.80,'Women',ha='right',fontsize=8.5,weight='semibold')
        fig.text(.977,.80,'Men',ha='right',fontsize=8.5,weight='semibold')
        lookup=paired.set_index('county_key')
        for index,k in enumerate(small_keys):
            y=.765-index*.025
            name=lookup.loc[k,'County']
            fig.text(.765,y,f'{small_lookup[k]:>2}  {name}',fontsize=8.5,weight='semibold',va='center')
            fig.text(.930,y,f"{lookup.loc[k,'female_pct']:.0f}%",ha='right',va='center',fontsize=8.5)
            fig.text(.977,y,f"{lookup.loc[k,'male_pct']:.0f}%",ha='right',va='center',fontsize=8.5)
        self._save_figure(fig,'county_prevalence_maps')
        return join

    @_themed
    def vif(self,table,threshold,name):
        self._require(table,['variable','vif'])
        vals=table.vif.to_numpy(float)
        if np.isnan(vals).any() or (vals<1-1e-8).any() or not np.isfinite(threshold) or threshold<=1:
            raise ValueError("VIF must be at least 1 and threshold must exceed 1")
        n=len(vals);fig=plt.figure(figsize=(10.5,max(5,.48*n+2.2)))
        ax=fig.add_axes([.31,.19,.57,.57])
        title=f"Collinearity diagnostics{' · '+self._sex() if self._sex() else ''}"
        stage='After VIF screening' if 'final' in str(name) else 'Before VIF screening'
        self._heading(fig,title,f"{stage} · {n} predictors · Review threshold: {threshold:g}",left=.06)
        finite=vals[np.isfinite(vals)]
        cap=max(threshold*1.3,float(finite.max())*1.08 if len(finite) else threshold*1.3)
        plotted=np.where(np.isfinite(vals),vals,cap)
        ax.axvspan(threshold,cap*1.12,color='#F7E2D8',alpha=.6,zorder=0)
        ax.axvline(threshold,color=self.FEMALE,linestyle=(0,(3,3)),linewidth=1)
        y=np.arange(n);colors=np.where(vals>threshold,self.FEMALE,self.MALE)
        ax.hlines(y,1,plotted,colors=colors,linewidth=2.2,alpha=.65)
        ax.scatter(plotted,y,c=colors,s=50,edgecolors=self.BG,linewidths=.7,zorder=3)
        for i,value in enumerate(vals):
            ax.text(plotted[i]+cap*.025,i,f"{value:.2f}" if np.isfinite(value) else '∞',
                    va='center',fontsize=9,color=self.INK,weight='semibold')
        ax.set_yticks(y,[self._label(v) for v in table.variable],fontsize=10)
        ax.set_ylim(n-.4,-.6);ax.set_xlim(.85,cap*1.14)
        ax.set_xlabel('Variance inflation factor',labelpad=12)
        self._style_axes(ax)
        self._footnote(fig,"Reference at VIF = 1; shaded region exceeds the configured review threshold.\n"
                       "VIF measures predictor redundancy, not substantive importance or model validity.",left=.06)
        self._save_figure(fig,name)

    @_themed
    def model_results(self,effects,predictions,include_effects=True):
        self._require(effects,['variable','AME_pp_per_reference_SD','ci_lower','ci_upper'])
        self._finite(effects,['AME_pp_per_reference_SD','ci_lower','ci_upper'])
        if ((effects.ci_lower>effects.AME_pp_per_reference_SD) | (effects.ci_upper<effects.AME_pp_per_reference_SD)).any():
            raise ValueError("Effect intervals must contain their point estimates")
        self._require(predictions,['County','observed_pct','predicted_pct'])
        self._finite(predictions,['observed_pct','predicted_pct'])
        if not predictions[['observed_pct','predicted_pct']].apply(lambda s:s.between(0,100)).all().all():
            raise ValueError("Model prevalence figures require the 0–100 scale")
        if include_effects:
            self._effect_figure(effects)
        self._fit_figure(predictions)

    def _effect_figure(self,effects):
        n=len(effects);fig=plt.figure(figsize=(11.8,max(5.3,.50*n+2.6)))
        gs=fig.add_gridspec(1,2,left=.28,right=.97,bottom=.14,top=.94,width_ratios=[3.3,1.65],wspace=.10)
        ax=fig.add_subplot(gs[0]);values=fig.add_subplot(gs[1],sharey=ax)
        self._heading(fig,f"Adjusted associations{' · '+self._sex() if self._sex() else ''}",
                      "Average marginal effects · Conditional model-based 95% intervals",left=.055)
        y=np.arange(n);color=self._color()
        for i in range(0,n,2):
            ax.axhspan(i-.5,i+.5,color='#F0F1ED',zorder=0)
            values.axhspan(i-.5,i+.5,color='#F0F1ED',zorder=0)
        ax.axvline(0,color='#8EA2A6',linewidth=1,linestyle=(0,(3,3)))
        ax.hlines(y,effects.ci_lower,effects.ci_upper,color=color,linewidth=2,alpha=.7)
        ax.scatter(effects.AME_pp_per_reference_SD,y,s=52,color=color,edgecolors=self.BG,linewidths=.8,zorder=3)
        ax.set_yticks(y,[self._label(v) for v in effects.variable],fontsize=10)
        ax.set_ylim(n-.5,-.7);self._style_axes(ax)
        bound=max(2,float(np.max(np.abs(effects[['ci_lower','ci_upper']].to_numpy())))*1.13)
        ax.set_xlim(-bound,bound);ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel('Percentage points per reference SD',labelpad=12)
        values.set_xlim(0,1);values.axis('off')
        values.text(.04,1.035,'AME  [95% interval]',transform=values.transAxes,fontsize=9,weight='semibold')
        for i,row in enumerate(effects.itertuples()):
            values.text(.04,i,f"{row.AME_pp_per_reference_SD:+.2f}  [{row.ci_lower:+.2f}, {row.ci_upper:+.2f}]",
                        va='center',fontsize=9,color=self.INK)
        self._footnote(fig,"Local average derivatives, not exact changes following a one-SD increase or causal effects.\n"
                       "Intervals condition on selection and do not incorporate survey-design or spatial uncertainty.",left=.055)
        self._save_figure(fig,'marginal_effects')

    def _fit_figure(self,predictions):
        observed=predictions.observed_pct.to_numpy(float);fitted=predictions.predicted_pct.to_numpy(float)
        residual=observed-fitted;mae=np.mean(abs(residual));rmse=np.sqrt(np.mean(residual**2))
        fig=plt.figure(figsize=(11.8,6.4))
        gs=fig.add_gridspec(1,2,left=.08,right=.96,top=.90,bottom=.13,wspace=.27)
        ax=fig.add_subplot(gs[0]);r=fig.add_subplot(gs[1]);color=self._color()
        self._heading(fig,f"How the model fits{' · '+self._sex() if self._sex() else ''}",
                      f"Training data only · {len(predictions)} counties · MAE {mae:.2f} pp · RMSE {rmse:.2f} pp",left=.065)
        self._style_axes(ax,grid='both');self._style_axes(r,grid='y')
        ax.plot([0,100],[0,100],color='#93A5A8',linestyle=(0,(3,3)),linewidth=1)
        ax.scatter(observed,fitted,s=39,color=color,alpha=.8,edgecolors=self.BG,linewidths=.6,zorder=3)
        ax.set(xlim=(-2,102),ylim=(-2,102),xlabel='Observed prevalence (%)',ylabel='Fitted prevalence (%)')
        ax.set_aspect('equal');ax.set_title('a  Observed and fitted',loc='left',pad=12)
        r.axhline(0,color='#93A5A8',linestyle=(0,(3,3)),linewidth=1)
        r.scatter(fitted,residual,s=39,color=color,alpha=.8,edgecolors=self.BG,linewidths=.6,zorder=3)
        bound=max(5,np.max(abs(residual))*1.30)
        r.set(xlim=(-2,102),ylim=(-bound,bound),xlabel='Fitted prevalence (%)',ylabel='Observed − fitted (pp)')
        r.set_title('b  Residual pattern',loc='left',pad=12)
        # Label up to three largest absolute residuals, with separated vertical slots.
        indices=np.argsort(abs(residual),kind='stable')[-min(3,len(residual)):]
        last_y=[]
        for index in indices[::-1]:
            x=fitted[index];y=residual[index];label_y=y
            while any(abs(label_y-yy)<bound*.17 for yy in last_y):
                label_y += -bound*.17 if y>0 else bound*.17
            label_y=float(np.clip(label_y,-bound*.88,bound*.88));last_y.append(label_y)
            left=x>55;target_x=x-12 if left else x+12
            r.annotate(str(predictions.County.iloc[index]),xy=(x,y),xytext=(target_x,label_y),
                       fontsize=8,color=self.INK,ha='right' if left else 'left',va='center',
                       arrowprops=dict(arrowstyle='-',color='#93A5A8',linewidth=.6))
        self._footnote(fig,"The dashed diagonal marks equality; the residual reference is zero. Labels identify the largest absolute residuals.\n"
                       "Training errors are descriptive fit measures, not cross-validated prediction performance.",left=.065)
        self._save_figure(fig,'observed_vs_fitted')

    @_themed
    def combined_marginal_effects(self, female_effects, male_effects=None,
                                  name='marginal_effects_combined'):
        """Paired AMEs and their supplied CIs; this is not a difference test.

        Supply two effects tables, or a {'female': table, 'male': table} mapping.
        SD and transformation metadata must agree for shared predictors.
        Absent predictors are labelled 'Not retained', never imputed as zero.
        """
        if male_effects is None:
            if not isinstance(female_effects,dict) or set(female_effects)!={'female','male'}:
                raise ValueError('Provide effects tables for both female and male models')
            tables={sex:table.copy() for sex,table in female_effects.items()}
        else:
            tables={'female':female_effects.copy(),'male':male_effects.copy()}
        for sex,table in tables.items():
            self._require(table,['variable','AME_pp_per_reference_SD','ci_lower','ci_upper',
                                 'reference_SD','transformation'])
            self._finite(table,['AME_pp_per_reference_SD','ci_lower','ci_upper','reference_SD'])
            if table.variable.duplicated().any() or (table.reference_SD<=0).any():
                raise ValueError(f'Invalid variable or reference SD in {sex} effects')
            if ((table.ci_lower>table.AME_pp_per_reference_SD) |
                (table.ci_upper<table.AME_pp_per_reference_SD)).any():
                raise ValueError('Effect intervals must contain their estimates')
            tables[sex]=table.set_index('variable')
        common=tables['female'].index.intersection(tables['male'].index)
        for key in common:
            f=tables['female'].loc[key];m=tables['male'].loc[key]
            if not np.isclose(f.reference_SD,m.reference_SD,rtol=1e-10,atol=1e-12) or f.transformation!=m.transformation:
                raise ValueError(f'Incompatible marginal-effect scales for {key}')
            if 'source_column' in f.index and 'source_column' in m.index and f.source_column!=m.source_column:
                raise ValueError(f'Incompatible source definitions for {key}')
        union=set(tables['female'].index)|set(tables['male'].index)
        order=[key for _,key,_,_ in CANDIDATES if key in union]
        order+=sorted(union-set(order))
        n=len(order)
        fig=plt.figure(figsize=(14.3,max(5.2,.59*n+1.25)))
        gs=fig.add_gridspec(1,3,left=.235,right=.985,bottom=.14,top=.91,
                           width_ratios=[3.45,1.85,1.85],wspace=.12)
        ax=fig.add_subplot(gs[0]);women=fig.add_subplot(gs[1],sharey=ax)
        men=fig.add_subplot(gs[2],sharey=ax)
        for panel in [ax,women,men]:
            for i in range(0,n,2):panel.axhspan(i-.48,i+.48,color='#F4F5F5',zorder=0)
            panel.set_ylim(n-.55,-.55)
        all_bounds=np.concatenate([t[['ci_lower','ci_upper']].to_numpy().ravel() for t in tables.values()])
        lo=min(0,float(all_bounds.min()));hi=max(0,float(all_bounds.max()));span=max(4,hi-lo)
        ax.set_xlim(lo-.08*span,hi+.08*span)
        ax.axvline(0,color='#7F939B',linewidth=.85,linestyle=(0,(3,3)))
        handles=[]
        for sex,label,color,marker,offset,panel in [
            ('female','Women',self.FEMALE,'o',-.14,women),
            ('male','Men',self.MALE,'s',.14,men)]:
            table=tables[sex]
            handles.append(Line2D([],[],marker=marker,color=color,lw=1.5,markersize=6,label=label))
            panel.axis('off');panel.set_xlim(0,1)
            panel.text(.04,1.035,f'{label}: AME [95% CI]',transform=panel.transAxes,
                       fontsize=9.2,weight='semibold',color=color,ha='left')
            for i,key in enumerate(order):
                if key not in table.index:
                    panel.text(.04,i,'Not retained',ha='left',va='center',fontsize=9,color='#89959A')
                    continue
                row=table.loc[key];y=i+offset
                ax.errorbar(row.AME_pp_per_reference_SD,y,
                            xerr=[[row.AME_pp_per_reference_SD-row.ci_lower],
                                  [row.ci_upper-row.AME_pp_per_reference_SD]],
                            fmt=marker,color=color,ms=5.6,mec='white',mew=.6,
                            elinewidth=1.5,capsize=2.5,capthick=1.2,zorder=3)
                panel.text(.04,i,f'{row.AME_pp_per_reference_SD:+.2f} [{row.ci_lower:+.2f}, {row.ci_upper:+.2f}]',
                           ha='left',va='center',fontsize=9,color='#273E48')
        ax.set_yticks(range(n),[self._label(key) for key in order],fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel('AME (percentage points per reference SD)',labelpad=12,fontsize=9.5)
        self._style_axes(ax)
        ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(0,1.015),ncol=2,
                  frameon=False,fontsize=9.4,borderaxespad=0,handlelength=2)
        self._save_figure(fig,name)

    @_themed
    def combined_vif(self, diagnostics, threshold=5., name='vif_combined'):
        """Initial nine-candidate versus final VIF, and explicit omission stage.

        Open markers show initial values; filled markers show final values.
        Red open markers show initial values for omitted predictors, with the
        omission status printed in the final-value column.
        No final VIF is invented for an omitted predictor. Initial-set VIF is
        diagnostic only and never changes the correlation-first selection rule.
        """
        self._require(diagnostics,['sex','variable','initial_vif','final_vif','status'])
        if set(diagnostics.sex)!={'female','male'}:
            raise ValueError('Combined VIF requires female and male diagnostics')
        if diagnostics.duplicated(['sex','variable']).any():
            raise ValueError('Duplicate combined VIF records')
        allowed={'retained','omitted_correlation','omitted_vif'}
        if not set(diagnostics.status).issubset(allowed):raise ValueError('Unknown VIF status')
        if not np.isfinite(threshold) or threshold<=1:raise ValueError('Invalid VIF threshold')
        vals=diagnostics.initial_vif.to_numpy(float)
        if np.isnan(vals).any() or (vals<1-1e-8).any():raise ValueError('Invalid initial VIF')
        final=diagnostics.loc[diagnostics.status.eq('retained'),'final_vif'].to_numpy(float)
        if not np.isfinite(final).all() or (final<1-1e-8).any():raise ValueError('Invalid retained VIF')
        if diagnostics.loc[~diagnostics.status.eq('retained'),'final_vif'].notna().any():
            raise ValueError('Omitted variables must not have a final-set VIF')
        union=set(diagnostics.variable)
        order=[key for _,key,_,_ in CANDIDATES if key in union]+sorted(union-{c[1] for c in CANDIDATES})
        if any(set(diagnostics.loc[diagnostics.sex.eq(sex),'variable'])!=union for sex in ['female','male']):
            raise ValueError('Both sexes must start with the same candidate set')
        finite=np.concatenate([vals[np.isfinite(vals)],final])
        cap=max(threshold*1.15,float(finite.max())*1.08)
        fig=plt.figure(figsize=(16,6.5))
        gs=fig.add_gridspec(1,4,left=.17,right=.985,bottom=.18,top=.86,
                            width_ratios=[2.5,1.45,2.5,1.45],wspace=.12)
        teal='#237C85';red='#C74B45';grey='#82969F'
        for panel,sex in enumerate(['female','male']):
            ax=fig.add_subplot(gs[2*panel]);tableax=fig.add_subplot(gs[2*panel+1],sharey=ax)
            data=diagnostics.loc[diagnostics.sex.eq(sex)].set_index('variable')
            for target in [ax,tableax]:
                for i in range(0,len(order),2):target.axhspan(i-.48,i+.48,color='#F2F5F5',zorder=0)
            ax.axvline(threshold,color='#9BABB0',ls=(0,(4,3)),lw=.9)
            for i,key in enumerate(order):
                item=data.loc[key];initial=float(item.initial_vif);removed=item.status!='retained'
                initial_x=initial if np.isfinite(initial) else cap
                color=red if removed else teal
                if not removed:
                    finalvalue=float(item.final_vif)
                    ax.plot([initial_x,finalvalue],[i-.10,i+.10],color=teal,lw=1.7,zorder=2)
                    ax.scatter(finalvalue,i+.10,s=38,c=teal,edgecolors='white',lw=.5,zorder=4)
                ax.scatter(initial_x,i-.10,s=38,facecolors='white',edgecolors=color,lw=1.5,zorder=3)
                tableax.text(.04,i,f'{initial:.2f}' if np.isfinite(initial) else '∞',
                             va='center',fontsize=9,color=color)
                text=f'{float(item.final_vif):.2f}' if not removed else 'Excluded¹' if item.status=='omitted_correlation' else 'Excluded²'
                tableax.text(.49,i,text,va='center',fontsize=8.5,color=color,weight='semibold')
            ax.set_yticks(range(len(order)),[self._label(k) for k in order],fontsize=9)
            if panel==1:ax.tick_params(labelleft=False)
            ax.set_ylim(len(order)-.5,-.65);ax.set_xlim(.8,cap*1.06)
            ax.xaxis.set_major_locator(MaxNLocator(5));self._style_axes(ax)
            ax.set_xlabel('Variance inflation factor',labelpad=10)
            ax.set_title('Women' if sex=='female' else 'Men',loc='left',fontsize=12,pad=20)
            tableax.set_xlim(0,1);tableax.axis('off')
            tableax.text(.04,1.035,'Initial',transform=tableax.transAxes,fontsize=9,weight='bold')
            tableax.text(.49,1.035,'Final',transform=tableax.transAxes,fontsize=9,weight='bold')
        handles=[Line2D([],[],marker='o',color=teal,mfc='white',lw=0,label='Initial set (open circle)'),
                 Line2D([],[],marker='o',color=teal,lw=0,label='Final retained set (filled circle)'),
                 Line2D([],[],marker='o',color=red,mfc='white',lw=0,label='Removed: initial VIF shown in red')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.57,.055),frameon=False,ncol=3,fontsize=8.5)
        fig.text(.57,.025,'¹ Excluded by correlation; ² excluded by VIF. Excluded variables have no final retained-set VIF.',
                 ha='center',fontsize=8,color='#566B75')
        self._save_figure(fig,name)

    @_themed
    def random_forest_comparison(self, metrics, importance, predictions):
        """Held-out comparison. Intervals show repeat ranges, never inferential CIs."""
        models = ['training_mean', 'fractional_logit_full', 'random_forest']
        names = ['Training-mean baseline', 'Fractional logit (all nine)', 'Random forest (all nine)']
        colors = ['#89989D', '#39759B', '#B06348']
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), sharex=True, sharey=True)
        for ax, sex in zip(axes, ['female', 'male']):
            for row, (model, color) in enumerate(zip(models, colors)):
                values = metrics.loc[metrics.sex.eq(sex) & metrics.model.eq(model), 'MAE_pp'].to_numpy()
                ax.axhspan(row-.35, row+.35, color=self.GRID, alpha=.32, zorder=0)
                ax.plot([values.min(), values.max()], [row, row], color=color, lw=2.5)
                offsets = np.linspace(-.11, .11, len(values))
                ax.scatter(values, row+offsets, s=22, color=color, alpha=.65, edgecolor='white', lw=.5)
                ax.scatter(values.mean(), row, s=65, color=color, edgecolor='white', zorder=4)
                ax.annotate(f'{values.mean():.2f}', (values.max(), row), xytext=(7, 0),
                            textcoords='offset points', va='center', fontsize=9, color=color)
            ax.set_yticks(range(len(models)), names)
            ax.set_ylim(2.65, -.65)
            ax.set_title('Women' if sex == 'female' else 'Men', loc='left')
            ax.set_xlabel('Held-out mean absolute error (percentage points)')
            self._style_axes(ax, grid='x')
            ax.margins(x=.2)
        fig.tight_layout()
        self._save_figure(fig, 'predictive_performance_comparison')

        # Forest held-out permutation importance; same variable order for both sexes.
        individual = importance.loc[importance.model.eq('random_forest') & importance.kind.eq('individual')]
        order = individual.groupby('variable').MAE_increase_pp.mean().sort_values(ascending=False).index.tolist()
        labels = {'log_population_density': 'Log population density', 'poverty_rate': 'Overall poverty',
                  'currently_employed': 'Current employment', 'completed_secondary': 'Completed secondary',
                  'household_size': 'Average household size', 'mass_media_access': 'Weekly media exposure',
                  'daily_internet_access': 'Daily internet use', 'gini_coefficient': 'Wealth Gini',
                  'acceptable_food_consumption': 'Acceptable food consumption'}
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.3), sharex=True, sharey=True)
        for ax, sex, color in zip(axes, ['female', 'male'], [self.FEMALE, self.MALE]):
            for row, key in enumerate(order):
                values = individual.loc[individual.sex.eq(sex) & individual.variable.eq(key), 'MAE_increase_pp'].to_numpy()
                ax.axhspan(row-.42, row+.42, color=self.GRID, alpha=.28, zorder=0)
                ax.plot([values.min(), values.max()], [row, row], color=color, lw=2)
                ax.scatter(values.mean(), row, color=color, s=42, edgecolor='white', zorder=3)
            ax.axvline(0, color=self.MUTED, ls='--', lw=.9)
            ax.set_yticks(range(len(order)), [labels.get(k, k) for k in order])
            ax.set_ylim(len(order)-.4, -.7)
            ax.set_title('Women' if sex == 'female' else 'Men', loc='left')
            ax.set_xlabel('Increase in held-out MAE (percentage points)')
            self._style_axes(ax, grid='x')
        fig.tight_layout()
        self._save_figure(fig, 'held_out_permutation_importance')

        fig, axes = plt.subplots(1, 2, figsize=(9.7, 4.8), sharex=True, sharey=True)
        for ax, sex in zip(axes, ['female', 'male']):
            for model, color, marker in [('fractional_logit_full', colors[1], 'o'), ('random_forest', colors[2], 's')]:
                # Mean of held-out predictions across repetitions, not a fitted-on-all model.
                subset = predictions.loc[predictions.sex.eq(sex) & predictions.model.eq(model)]
                county = subset.groupby('county_key').agg(observed=('observed', 'first'), predicted=('predicted', 'mean'))
                ax.scatter(100*county.observed, 100*county.predicted, c=color, marker=marker,
                           s=28, alpha=.72, edgecolor='white', lw=.4,
                           label='Fractional logit' if model == 'fractional_logit_full' else 'Random forest')
            ax.plot([0, 100], [0, 100], '--', color=self.MUTED, lw=.8)
            ax.set(xlim=(0, 100), ylim=(0, 100), xlabel='Observed prevalence (%)')
            ax.set_title('Women' if sex == 'female' else 'Men', loc='left')
            ax.set_aspect('equal')
            self._style_axes(ax)
        axes[0].set_ylabel('Mean held-out prediction (%)')
        axes[1].legend(frameon=False, fontsize=8, loc='upper left')
        fig.tight_layout()
        self._save_figure(fig, 'held_out_observed_predicted')
