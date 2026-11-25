"""Nested county CV: random forest regression versus full fractional logit.

Use ModelData.source_data: fixed source transformations, NOT globally scaled
ModelData.data. Every predictive scaling operation is fitted on training rows.
All nine predictors are used; no outcome-based selection or VIF exclusions.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import platform
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
import statsmodels
import statsmodels.api as sm

from .plotter import Plotter


@dataclass(frozen=True)
class RandomForestConfig:
    seed: int = 20261125
    outer_splits: int = 5
    repeats: int = 5
    inner_splits: int = 3
    n_estimators: int = 300
    permutation_repeats: int = 10
    n_jobs: int = 2
    # Small, deliberately restricted grid for 47 counties per sex.
    max_depths: tuple = (2, 4)
    min_samples_leaves: tuple = (4, 8)
    max_features: float = 1.0

    def __post_init__(self):
        for key in ('outer_splits', 'inner_splits'):
            if getattr(self, key) < 2:
                raise ValueError(f'{key} must be at least 2')
        for key in ('repeats', 'n_estimators', 'permutation_repeats'):
            if getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive')
        if any(d is not None and d < 1 for d in self.max_depths):
            raise ValueError('Tree depths must be positive or None')
        if not self.max_depths or not self.min_samples_leaves:
            raise ValueError('Tuning grid cannot be empty')
        if any(v < 1 for v in self.min_samples_leaves):
            raise ValueError('Minimum leaf sizes must be positive')
        if not 0 < self.max_features <= 1:
            raise ValueError('max_features must be in (0, 1]')


class ComplementaryRandomForest:
    """Regression on county proportions; equal county weights throughout.

    Output is output_directory/random_forest/{female,male,comparison,figures}.
    Repeated CV distributions are descriptive, NOT independent-sample CIs.
    County folds match between sexes; each sex is fitted separately.
    """
    RF = 'random_forest'
    FL = 'fractional_logit_full'
    BASE = 'training_mean'

    def __init__(self, output_directory, config=None):
        self.output = Path(output_directory) / 'random_forest'
        self.config = config or RandomForestConfig()

    def forest(self, settings, seed):
        return RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=settings['max_depth'],
            min_samples_leaf=settings['min_samples_leaf'],
            max_features=self.config.max_features,
            random_state=int(seed), n_jobs=self.config.n_jobs,
            criterion='squared_error', bootstrap=True)

    @staticmethod
    def fractional_fit(x, y):
        scaler = StandardScaler().fit(x)
        a = sm.add_constant(scaler.transform(x), has_constant='add')
        if len(y) <= a.shape[1] or np.linalg.matrix_rank(a) != a.shape[1]:
            raise ValueError('Training fractional-logit design is rank deficient or too small')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit = sm.GLM(y, a, family=sm.families.Binomial()).fit(
                maxiter=500, tol=1e-10)
        if not fit.converged or not np.isfinite(fit.params).all():
            raise RuntimeError('Training fractional-logit comparator did not converge')
        mu = expit(a @ np.asarray(fit.params))
        score = a.T @ (y - mu)
        if np.max(np.abs(score)) > 1e-6:
            raise RuntimeError('Training fractional-logit quasi-score check failed')
        return scaler, np.asarray(fit.params), [str(v.message) for v in caught]

    @staticmethod
    def fractional_predict(fitted, x):
        scaler, beta, _ = fitted
        return expit(sm.add_constant(scaler.transform(x), has_constant='add') @ beta)

    @staticmethod
    def metrics(y, pred):
        error = 100 * (np.asarray(pred) - np.asarray(y))
        return {'MAE_pp': float(np.mean(np.abs(error))),
                'RMSE_pp': float(np.sqrt(np.mean(error ** 2)))}

    def _tune(self, x, y, seed, repeat, fold):
        cv = list(KFold(self.config.inner_splits, shuffle=True,
                        random_state=seed).split(x))
        rows = []
        grid = list(ParameterGrid({'max_depth': self.config.max_depths,
                                  'min_samples_leaf': self.config.min_samples_leaves}))
        # The same inner splits and estimator seeds are used across settings.
        for index, settings in enumerate(grid):
            prediction = np.full(len(y), np.nan)
            for inner, (train, valid) in enumerate(cv):
                fit = self.forest(settings, seed + inner)
                fit.fit(x[train], y[train])
                prediction[valid] = fit.predict(x[valid])
            rows.append(dict(repeat=repeat, fold=fold, setting=index,
                             **settings, **self.metrics(y, prediction)))
        best = min(range(len(rows)), key=lambda i: (rows[i]['MAE_pp'], i))
        for i, row in enumerate(rows):
            row['selected'] = i == best
        return grid[best], rows

    def _importance(self, models, x, y, features, repeat, fold, seed):
        # Fixed a priori group, identical across sexes. Joint permutation retains
        # within-group relationships but breaks association with other variables.
        grouped = [k for k in ('poverty_rate', 'household_size', 'mass_media_access')
                   if k in features]
        groups = [(key, [i], 'individual') for i, key in enumerate(features)]
        if len(grouped) > 1:
            groups.append(('poverty_household_media',
                           [features.index(k) for k in grouped], 'joint_group'))
        rng = np.random.default_rng(seed)
        permutations = [rng.permutation(len(y))
                        for _ in range(self.config.permutation_repeats)]
        # Batch all shuffled test sets: identical permutations, far less
        # thread-dispatch overhead than predicting one tiny county set at a time.
        shuffled_sets, metadata = [], []
        for label, columns, kind in groups:
            for iteration, order in enumerate(permutations):
                shuffled = x.copy()
                shuffled[:, columns] = x[order][:, columns]
                shuffled_sets.append(shuffled)
                metadata.append((label, kind, iteration))
        batch = np.concatenate(shuffled_sets, axis=0)
        rows = []
        for model, predict in models.items():
            baseline = self.metrics(y, predict(x))['MAE_pp']
            batch_predictions = np.asarray(predict(batch)).reshape(len(metadata), len(y))
            for (label, kind, iteration), pred in zip(metadata, batch_predictions):
                increase = self.metrics(y, pred)['MAE_pp'] - baseline
                rows.append(dict(repeat=repeat, fold=fold, model=model,
                                 variable=label, kind=kind, permutation=iteration,
                                 n_test=len(y), MAE_increase_pp=increase))
        return rows

    def run(self, prepared):
        self.output.mkdir(parents=True, exist_ok=True)
        status = self.output / 'run_status.json'
        status.write_text(json.dumps({'status': 'running', 'config': asdict(self.config)}, indent=2))
        try:
            result = self._run(prepared)
            status.write_text(json.dumps({'status': 'completed', 'config': asdict(self.config)}, indent=2))
            return result
        except Exception as exc:
            status.write_text(json.dumps({'status': 'failed', 'error': str(exc),
                                         'config': asdict(self.config)}, indent=2))
            raise

    def _run(self, prepared):
        features = list(prepared.features)
        data = prepared.source_data.copy()
        if len(features) != 9 or len(set(features)) != 9:
            raise ValueError('The complementary comparison requires all nine predictors')
        if set(data.sex) != {'female', 'male'}:
            raise ValueError('Both female and male county datasets are required')
        frames = {}
        for sex in ('female', 'male'):
            frame = data.loc[data.sex.eq(sex)].sort_values('county_key').reset_index(drop=True)
            if frame.county_key.duplicated().any():
                raise ValueError(f'{sex}: duplicate county identifiers')
            matrix = frame[features + ['y']].to_numpy(dtype=float)
            if not np.isfinite(matrix).all() or not frame.y.between(0, 1).all():
                raise ValueError(f'{sex}: invalid county proportions or predictors')
            if frame.y.nunique() < 2:
                raise ValueError(f'{sex}: outcome is constant')
            frames[sex] = frame
        keys = frames['female'].county_key.tolist()
        if keys != frames['male'].county_key.tolist():
            raise ValueError('County sets must match between sexes')
        n = len(keys)
        if self.config.outer_splits > n:
            raise ValueError('More outer folds than counties')
        plans, assignments = [], []
        for repeat in range(self.config.repeats):
            for fold, (train, test) in enumerate(KFold(
                    self.config.outer_splits, shuffle=True,
                    random_state=self.config.seed + repeat).split(np.arange(n))):
                if len(train) <= len(features) + 1 or self.config.inner_splits > len(train):
                    raise ValueError('Insufficient training counties for the specified CV')
                plans.append((repeat, fold, train, test))
                for i in test:
                    assignments.append({'county_key': keys[i], 'repeat': repeat, 'fold': fold})
        pd.DataFrame(assignments).to_csv(self.output / 'county_fold_assignments.csv', index=False)
        comparison = self.output / 'comparison'
        comparison.mkdir(exist_ok=True)
        all_predictions, all_metrics, all_importance = [], [], []
        for sex, frame in frames.items():
            folder = self.output / sex
            folder.mkdir(exist_ok=True)
            frame[['County', 'county_key', 'y'] + features].to_csv(
                folder / 'comparison_input_transformed_unscaled.csv', index=False)
            x, y = frame[features].to_numpy(float), frame.y.to_numpy(float)
            predictions, tuning, importance, fits, scaling = [], [], [], [], []
            for task, (repeat, fold, train, test) in enumerate(plans, 1):
                seed = self.config.seed + repeat * 1000 + fold * 20
                print(f'      {sex}: outer fit {task}/{len(plans)} '
                      f'(repeat {repeat + 1}, fold {fold + 1})', flush=True)
                best, rows = self._tune(x[train], y[train], seed, repeat, fold)
                tuning.extend(rows)
                forest = self.forest(best, seed)
                forest.fit(x[train], y[train])
                fractional = self.fractional_fit(x[train], y[train])
                scaler, beta, caught = fractional
                for j, variable in enumerate(features):
                    scaling.append(dict(repeat=repeat, fold=fold, variable=variable,
                                        training_mean=scaler.mean_[j],
                                        training_SD=scaler.scale_[j], coefficient=beta[j+1]))
                fits.append(dict(repeat=repeat, fold=fold, n_train=len(train), n_test=len(test),
                                 intercept=beta[0], converged=True, warnings=' | '.join(caught), **best))
                models = {self.RF: forest.predict,
                          self.FL: lambda values: self.fractional_predict(fractional, values)}
                for model in (self.RF, self.FL, self.BASE):
                    pred = (np.full(len(test), y[train].mean()) if model == self.BASE
                            else models[model](x[test]))
                    if not np.isfinite(pred).all() or ((pred < 0) | (pred > 1)).any():
                        raise RuntimeError('Nonfinite or out-of-range prediction')
                    for i, value in zip(test, pred):
                        predictions.append(dict(sex=sex, repeat=repeat, fold=fold,
                                                county_key=keys[i], County=frame.County.iloc[i],
                                                model=model, observed=y[i], predicted=float(value)))
                importance.extend(self._importance(models, x[test], y[test], features,
                                                   repeat, fold, seed + 10))
            prediction = pd.DataFrame(predictions)
            repeat_metrics = []
            for (repeat, model), group in prediction.groupby(['repeat', 'model']):
                if len(group) != n or group.county_key.nunique() != n:
                    raise AssertionError('Each county must be predicted exactly once per repeat/model')
                repeat_metrics.append(dict(sex=sex, repeat=repeat, model=model,
                                           **self.metrics(group.observed, group.predicted)))
            repeat_metrics = pd.DataFrame(repeat_metrics)
            imp = pd.DataFrame(importance)
            imp['sex'] = sex
            # Weight held-out fold importance by county count, not equally by fold.
            imp['weighted'] = imp.MAE_increase_pp * imp.n_test
            repeat_imp = imp.groupby(['sex', 'repeat', 'model', 'variable', 'kind'], as_index=False).agg(
                weighted_sum=('weighted', 'sum'), total_weight=('n_test', 'sum'))
            repeat_imp['MAE_increase_pp'] = repeat_imp.weighted_sum / repeat_imp.total_weight
            repeat_imp = repeat_imp.drop(columns=['weighted_sum', 'total_weight'])
            prediction.to_csv(folder / 'out_of_fold_predictions.csv', index=False)
            repeat_metrics.to_csv(folder / 'repeat_metrics.csv', index=False)
            pd.DataFrame(tuning).to_csv(folder / 'inner_cv_tuning.csv', index=False)
            pd.DataFrame(fits).to_csv(folder / 'outer_fit_diagnostics.csv', index=False)
            pd.DataFrame(scaling).to_csv(folder / 'training_scaling_and_coefficients.csv', index=False)
            imp.drop(columns='weighted').to_csv(folder / 'held_out_permutation_draws.csv', index=False)
            repeat_imp.to_csv(folder / 'repeat_permutation_importance.csv', index=False)
            all_predictions.append(prediction)
            all_metrics.append(repeat_metrics)
            all_importance.append(repeat_imp)
        predictions = pd.concat(all_predictions, ignore_index=True)
        metrics = pd.concat(all_metrics, ignore_index=True)
        importance = pd.concat(all_importance, ignore_index=True)
        summary = metrics.groupby(['sex', 'model'], as_index=False).agg(
            MAE_pp_mean=('MAE_pp', 'mean'), MAE_pp_SD=('MAE_pp', 'std'),
            MAE_pp_min=('MAE_pp', 'min'), MAE_pp_max=('MAE_pp', 'max'),
            RMSE_pp_mean=('RMSE_pp', 'mean'), RMSE_pp_SD=('RMSE_pp', 'std'),
            RMSE_pp_min=('RMSE_pp', 'min'), RMSE_pp_max=('RMSE_pp', 'max'))
        imp_summary = importance.groupby(['sex', 'model', 'variable', 'kind'], as_index=False).agg(
            MAE_increase_pp_mean=('MAE_increase_pp', 'mean'),
            MAE_increase_pp_SD=('MAE_increase_pp', 'std'),
            MAE_increase_pp_min=('MAE_increase_pp', 'min'),
            MAE_increase_pp_max=('MAE_increase_pp', 'max'))
        differences = []
        for (sex, repeat), rows in metrics.groupby(['sex', 'repeat']):
            rows = rows.set_index('model')
            differences.append(dict(sex=sex, repeat=repeat,
                MAE_difference_RF_minus_FL_pp=rows.loc[self.RF, 'MAE_pp'] - rows.loc[self.FL, 'MAE_pp'],
                RMSE_difference_RF_minus_FL_pp=rows.loc[self.RF, 'RMSE_pp'] - rows.loc[self.FL, 'RMSE_pp']))
        summary.to_csv(comparison / 'predictive_performance_summary.csv', index=False)
        metrics.to_csv(comparison / 'repeat_metrics.csv', index=False)
        pd.DataFrame(differences).to_csv(comparison / 'paired_model_differences.csv', index=False)
        imp_summary.to_csv(comparison / 'permutation_importance_summary.csv', index=False)
        predictions.to_csv(comparison / 'out_of_fold_predictions.csv', index=False)
        versions = {'python': platform.python_version(), 'numpy': np.__version__,
                    'pandas': pd.__version__, 'scipy': scipy.__version__,
                    'scikit_learn': sklearn.__version__, 'statsmodels': statsmodels.__version__}
        (self.output / 'analysis_config.json').write_text(json.dumps(
            {'config': asdict(self.config), 'features': features, 'n_counties_per_sex': n,
             'versions': versions, 'tuning_metric': 'inner out-of-fold MAE',
             'weighting': 'equal counties', 'scaling': 'training-only, fractional comparator',
             'split_unit': 'county; matched across sexes; random, not spatial blocks'}, indent=2))
        (self.output / 'interpretation.txt').write_text(
            'All nine predictors; no correlation or VIF screening in this comparison.\n'
            'MAE and RMSE are in percentage points; lower is better.\n'
            'Each repeat scores one out-of-fold prediction per county and model.\n'
            'Paired differences: negative means lower error for random forest.\n'
            'Repeat SDs and ranges describe partition variability, NOT confidence intervals.\n'
            'Permutation values are held-out MAE increases; negatives are retained.\n'
            'Correlated variables can share importance; the fixed poverty/household/media group\n'
            'is permuted jointly with one common row order. Group importance is not additive.\n'
            'Neither importance nor cross-sex rankings establish causal or sex effects.\n'
            'Random county validation does not assess future or spatially independent prediction.\n'
            'Survey sampling uncertainty is not propagated. Small county sample limits stability.\n'
            'Only read numerical files when run_status.json says completed.\n')
        Plotter(self.output / 'figures').random_forest_comparison(metrics, importance, predictions)
        return {'performance': summary, 'importance': imp_summary, 'output_dir': self.output}
