"""Fractional-logistic models after sex-specific correlation and VIF screening."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t
import statsmodels.api as sm
from .vif import calculate_vif


def gradient(function, beta):
    out = np.zeros(len(beta))
    for i in range(len(beta)):
        h = 1e-5 * max(1., abs(beta[i]))
        plus, minus = beta.copy(), beta.copy()
        plus[i] += h; minus[i] -= h
        out[i] = (function(plus) - function(minus)) / (2 * h)
    return out


class FractionalLogit:
    """Separate-sex fractional logit; HC1 quasi-score sandwich, t(N-K).

    Intervals are conditional on the selected variable set and do not account
    for screening uncertainty, spatial dependence or KDHS sampling design.
    No additional p-value elimination is performed.
    """
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def fit(self, data, features, sex, dictionary=None):
        folder = self.output_dir / sex
        folder.mkdir(parents=True, exist_ok=True)
        # A failed rerun must not leave older successful numerical results visible.
        for name in ['coefficients.csv','robust_covariance.csv','marginal_effects.csv',
                     'predictions.csv','fit_diagnostics.csv','leverage.csv','interpretation.txt']:
            (folder/name).unlink(missing_ok=True)
        features = list(features)
        x = pd.DataFrame({"const": np.ones(len(data))})
        for key in features:
            x[key] = data[key].to_numpy()
        matrix, y = x.to_numpy(dtype=float), data.y.to_numpy(dtype=float)
        n, k = matrix.shape
        if len(features) != len(set(features)):
            raise ValueError('Duplicate model variables')
        if not np.isfinite(matrix).all() or not np.isfinite(y).all() or not ((y >= 0) & (y <= 1)).all():
            raise ValueError('Invalid fractional-logistic inputs')
        rank = np.linalg.matrix_rank(matrix)
        starting = dict(sex=sex, n_counties=n, n_parameters=k, design_rank=int(rank),
                        residual_df=n-k, model=self.__class__.__name__)
        (folder/'fit_status.json').write_text(json.dumps(dict(**starting,status='running'),indent=2))
        pd.DataFrame(dict(variable=features)).to_csv(folder / "model_variables.csv", index=False)
        calculate_vif(data, features).to_csv(folder / "model_vif.csv", index=False)
        if n <= k or rank != k:
            (folder / "fit_status.json").write_text(json.dumps(dict(
                **starting, status="not_estimable", reason="Rank deficiency or insufficient residual degrees of freedom"), indent=2))
            return None
        if not np.isfinite(matrix).all() or not np.isfinite(y).all() or not ((y>=0)&(y<=1)).all():
            raise ValueError("Invalid fractional-logistic inputs")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=300, tol=1e-10)
        beta = np.asarray(fit.params)
        mu = expit(matrix @ beta)
        score = matrix * (y-mu)[:,None]
        bread = matrix.T @ ((mu*(1-mu))[:,None]*matrix)
        if not fit.converged or np.max(abs(score.sum(axis=0)))>1e-6 or np.linalg.matrix_rank(bread)!=k:
            raise RuntimeError(f"{sex}: model did not solve identifiable quasi-score equations")
        inverse = np.linalg.inv(bread)
        covariance = n/(n-k) * inverse @ (score.T @ score) @ inverse
        covariance = (covariance+covariance.T)/2
        if np.linalg.eigvalsh(covariance).min() < -1e-7:
            raise RuntimeError("Invalid robust covariance")
        se=np.sqrt(np.maximum(0,np.diag(covariance)))
        critical=t.ppf(.975,n-k)
        coefficients=pd.DataFrame(dict(variable=x.columns,coefficient=beta,HC1_se=se,
            ci_lower=beta-critical*se,ci_upper=beta+critical*se,
            p_value=2*t.sf(abs(beta/se),n-k),inference_df=n-k))
        coefficients.to_csv(folder/'coefficients.csv',index=False)
        pd.DataFrame(covariance,index=x.columns,columns=x.columns).to_csv(folder/'robust_covariance.csv')
        effects=[]
        for i,key in enumerate(features,1):
            def effect(b, i=i):
                p=expit(matrix@b)
                return float(100*np.mean(p*(1-p))*b[i])
            value=effect(beta); g=gradient(effect,beta)
            sd=np.sqrt(max(0,float(g@covariance@g)))
            effects.append(dict(variable=key,AME_pp_per_reference_SD=value,HC1_se=sd,
                                ci_lower=value-critical*sd,ci_upper=value+critical*sd,
                                p_value=2*t.sf(abs(value/sd),n-k),inference_df=n-k))
        effects=pd.DataFrame(effects)
        if dictionary is not None:
            effects=effects.merge(dictionary[['feature','source_column','reference_SD','transformation']],
                                  left_on='variable',right_on='feature',validate='one_to_one').drop(columns='feature')
        effects.to_csv(folder/'marginal_effects.csv',index=False)
        eta=matrix@beta
        eta_se=np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',matrix,covariance,matrix)))
        predictions=data[['County','county_key']].copy()
        predictions['observed_pct']=100*y
        predictions['predicted_pct']=100*mu
        predictions['mean_ci_lower_pct']=100*expit(eta-critical*eta_se)
        predictions['mean_ci_upper_pct']=100*expit(eta+critical*eta_se)
        predictions['residual_pct']=100*(y-mu)
        predictions.to_csv(folder/'predictions.csv',index=False)
        leverage=np.sum(matrix*(matrix@inverse),axis=1)*mu*(1-mu)
        centred=matrix[:,1:]-matrix[:,1:].mean(axis=0)
        normalized=np.column_stack([np.ones(n),centred/centred.std(axis=0,ddof=0)])
        diagnostics=dict(**starting,status='converged',max_abs_score=float(abs(score.sum(axis=0)).max()),
                         condition_number=float(np.linalg.cond(matrix)),training_MAE_pp=float(100*abs(y-mu).mean()),
                         within_sex_standardized_condition_number=float(np.linalg.cond(normalized)),
                         training_RMSE_pp=float(100*np.sqrt(np.mean((y-mu)**2))),max_leverage=float(leverage.max()),
                         counties_per_predictor=float(n/max(1,k-1)),
                         warnings=' | '.join(str(w.message) for w in caught))
        pd.DataFrame([diagnostics]).to_csv(folder/'fit_diagnostics.csv',index=False)
        (folder/'fit_status.json').write_text(json.dumps(diagnostics,indent=2))
        predictions[['County','county_key']].assign(leverage=leverage).to_csv(folder/'leverage.csv',index=False)
        (folder/'interpretation.txt').write_text(
            'Separate-sex fractional logit, not binary outcomes or invented trial counts.\n'
            'Predictors standardized using the common full-data reference in the dictionary.\n'
            'HC1 sandwich with N/(N-K) correction; t(N-K) is a working finite-sample approximation.\n'
            'P-values/intervals conditional on screening, not selection-adjusted or survey-design inference.\n'
            'AMEs are local derivatives per reference SD, not exact finite changes or causal effects.\n'
            'Different sex-specific retained sets imply different adjustment; do not infer gender differences\n'
            'from significance in one model but not the other.\n'
            'Model selection remains exploratory; compare both screening stages.\n'
            'All source-quality flags remain unresolved; no source values were corrected.\n')
        return dict(result=fit,design=x,covariance=covariance,effects=effects,diagnostics=diagnostics)
