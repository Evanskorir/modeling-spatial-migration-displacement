# Internal migration prevalence in Kenya: County-level patterns and socioeconomic associations among women and men

## Overview

This project examines geographical variation and socioeconomic associations in internal migration prevalence among women and men across Kenya's counties. It combines published 2022 Kenya Demographic and Health Survey (KDHS) estimates with official county indicators.

The analysis uses sex-specific fractional-logit models, correlation and multicollinearity screening, and a common full nine-variable specification. Complementary random forest regression evaluates held-out predictive performance against full fractional logit and a training-mean baseline.

The analysis is implemented in `redesign/`.
## Study outcome

The outcome is the proportion of respondents aged 15–49 born elsewhere in Kenya than their current place of residence, summarized by county and sex. Estimates come from KDHS Tables 3.15.1C and 3.15.2C.

- Visitors are excluded from the eligible respondent population.
- Respondents born outside Kenya are excluded from the numerator.
- The denominator includes all eligible respondents, rather than only those born in Kenya.
- Counties are the units of aggregation; recorded moves need not have crossed county boundaries.
- The measure does not identify migration dates, intermediate destinations, return moves, recent inflows or net migration.

Each sex-specific dataset contains one observation for each of Kenya's 47 counties. Published KDHS estimates incorporate survey weights. Subsequent descriptive summaries and models give equal weight to counties.


## Data sources

The analysis combines:

- KDHS 2022 Volume 1: migration, employment, education, media exposure, internet use, wealth Gini and food consumption.
- Kenya Population and Housing Census 2019, Volume I: population density and average household size.
- Gross County Product 2024 report: overall poverty estimates referring to 2022.

### Input files

| File in `data/` | Contents |
|---|---|
| `data_female.xls` | County outcome and explanatory variables for women |
| `data_male.xls` | County outcome and explanatory variables for men |
| `kenya_counties.geojson` | County boundaries for mapping |

The loader also accepts `.xlsx` workbooks when their filenames are specified through the command line.

### Required workbook columns

Use the following exact column names. Input percentages must be supplied on the **0–100 scale**.

| Column | Definition |
|---|---|
| `County` | County name |
| `Born_in_Kenya_but_outside` | Percentage born elsewhere in Kenya than current residence |
| `Population Density` | Persons per square kilometre |
| `Poverty Rate` | Percentage below the overall poverty line |
| `Currently_employed` | Percentage currently employed, by sex |
| `Completed_secondary` | Percentage in the completed-secondary attainment category, by sex |
| `Average Household Size` | Mean persons per household |
| `Mass_Media_Access` | Percentage exposed to newspapers, radio or television at least weekly, by sex |
| `Daily_Internet_Access` | Percentage using the internet almost daily in the preceding month among internet users in the preceding 12 months, by sex |
| `Gini_coefficient` | Wealth concentration on the 0–1 scale |
| `Acceptable_Food_Consumption_Score` | Percentage of households with valid food consumption scores classified as acceptable |


## Analysis workflow

1. **Data validation and preparation:** harmonize county identifiers, verify matched coverage, check selected inputs and export source and quality checks.
2. **Descriptive analysis:** summarize explanatory variables and migration prevalence, and produce county maps and distributions.
3. **Correlation screening:** separately identify redundant explanatory variables for women and men.
4. **Correlation-screened estimation:** fit fractional-logit models using the correlation-retained variables.
5. **VIF assessment:** assess remaining multicollinearity and fit the final screened specifications.
6. **Full-specification estimation:** fit all nine explanatory variables for each sex.
7. **Complementary prediction:** compare random forest regression, full fractional logit and a training-mean baseline using nested county cross-validation.

### Transformations and scaling

Population density is log-transformed, and percentages are converted to proportions. Household size and wealth Gini retain their original scales before standardization.

For association models, explanatory variables are standardized using a common mean and population standard deviation across the county–sex observations.

Predictive validation uses separate training-only standardization for the fractional-logit comparator. Forests use transformed variables without standardization.

### Correlation and multicollinearity screening

Screening is performed separately for women and men.

For the pair with the largest absolute Pearson correlation above 0.80, the variable appearing later in the following fixed order is excluded:

1. Population density
2. Overall poverty
3. Current employment
4. Completed secondary education
5. Average household size
6. Weekly media exposure
7. Daily internet use
8. Wealth Gini coefficient
9. Acceptable food consumption

The procedure continues until all remaining pairs satisfy the correlation threshold.

VIF assessment then specifies iterative removal of the variable with the highest VIF above 5, followed by recalculation.

### Fractional-logit estimation

The proportional outcome is modelled using Bernoulli quasi-maximum likelihood, separately for each sex.

The analysis reports:

- Regression coefficients.
- HC1 robust standard errors.
- Confidence intervals using a working Student's t reference.
- Average marginal effects in percentage points per common reference standard deviation.
- Convergence, rank, residual and leverage diagnostics.
- In-sample MAE and RMSE.

### Complementary random forest regression

The predictive stage uses regression on county proportions. Both fitted models use all nine explanatory variables without correlation or VIF exclusions within this comparison.

| Setting | Default |
|---|---|
| Outer validation | Five county folds, repeated five times |
| Inner validation | Three folds within each outer training sample |
| Trees per forest | 300 |
| Maximum depth grid | 2 or 4 |
| Minimum terminal-node size grid | 4 or 8 |
| Features eligible at each split | All nine |
| Tree sampling | Bootstrap samples |
| Tuning criterion | Inner out-of-fold MAE |
| Permutations per variable or group | 10 |
| Random seed | 20261125 |

## Code organization

| Module in `redesign/` | Responsibility |
|---|---|
| `data_loader.py` | Read workbooks and validate counties, outcomes and source compositions |
| `variables.py` | Define explanatory variables, transformations and screening order |
| `model_data.py` | Prepare transformed and commonly standardized model data |
| `descriptives.py` | Export descriptive summaries and coordinate plotting |
| `correlation.py` | Perform sex-specific correlation screening |
| `vif.py` | Calculate VIFs and export screening diagnostics |
| `fractional_logit.py` | Estimate models, robust intervals, marginal effects and diagnostics |
| `logistic_after_correlation.py` | Fit correlation-screened models |
| `logistic_retained.py` | Fit final screened models |
| `logistic_full.py` | Fit full nine-variable models |
| `random_forest_analysis.py` | Run nested predictive comparison and permutation importance |
| `plotter.py` | Centralize figure rendering and PDF/PNG export |
| `runner.py` | Coordinate analysis stages and record run status |
| `main_redesign.py` | Provide the command-line entry point |

## Installation

The shared workflow was validated with Python 3.12. Python 3.11 or newer is recommended.

Install dependencies using the `requirements.txt` supplied with the redesign package:

```bash
python -m pip install -r requirements.txt
```

Core dependencies are:

- `numpy`
- `pandas`
- `scipy`
- `statsmodels`
- `scikit-learn`
- `matplotlib`
- `openpyxl`
- `xlrd`

Use a dedicated environment and retain installed package versions with published outputs.

## Running the analysis

Run commands from the project root containing `data/` and `redesign/`.

### Complete workflow

```bash
python redesign/main_redesign.py --project-dir .
```

Alternatively:

```bash
python -m redesign.main_redesign --project-dir .
```

## Outputs

The default output root is:

```text
output/redesign/nine_variables_publication/
```

| Path relative to the output root | Contents |
|---|---|
| `descriptive/tables/` | Variable summaries, county prevalence, sex differences, quality checks and source hashes |
| `descriptive/model_setup/` | Prepared model data and common reference scales |
| `descriptive/figures/` | Descriptive figures and maps when enabled |
| `correlation/female/` | Women's correlation matrices, screening decisions and figures |
| `correlation/male/` | Men's correlation matrices, screening decisions and figures |
| `vif/female/` | Women's VIF diagnostics and retained variables |
| `vif/male/` | Men's VIF diagnostics and retained variables |
| `vif/figures/` | Combined VIF figure |
| `logistic_after_correlation/` | Correlation-screened estimates and figures |
| `logistic_after_vif/` | Final screened estimates and figures |
| `logistic_full/` | Full nine-variable estimates and figures |
| `random_forest/female/` | Women's predictive outputs |
| `random_forest/male/` | Men's predictive outputs |
| `random_forest/comparison/` | Performance and importance comparisons |
| `random_forest/figures/` | Predictive comparison figures |
| `random_forest/analysis_config.json` | Settings, predictor names and package versions |
| `model_summary.csv` | In-sample association-model diagnostics |
| `retained_set_comparison.csv` | Final variable inclusion by sex |
| `run_status.json` | Status of the requested workflow |

## Methodological references

- Papke, L. E., and Wooldridge, J. M. (1996). Econometric methods for fractional response variables with an application to 401(k) plan participation rates. *Journal of Applied Econometrics*, 11(6), 619–632.
- O'Brien, R. M. (2007). A caution regarding rules of thumb for variance inflation factors. *Quality & Quantity*, 41, 673–690. https://doi.org/10.1007/s11135-006-9018-6
- Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324