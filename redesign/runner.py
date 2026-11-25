"""County inference workflow plus full-specification predictive comparison."""
from pathlib import Path
import json
from datetime import datetime,timezone
import pandas as pd
from .data_loader import MigrationDataLoader
from .descriptives import MigrationDescriptives
from .model_data import ModelData
from .correlation import CorrelationAnalysis
from .vif import VIFAnalysis
from .logistic_retained import LogisticRetained
from .logistic_after_correlation import LogisticAfterCorrelation
from .plotter import Plotter
from .logistic_full import LogisticFull
from .random_forest_analysis import ComplementaryRandomForest, RandomForestConfig


class AnalysisRunner:
    def __init__(self,project_dir,data_dir=None,output_dir=None,correlation_threshold=.8,
                 vif_threshold=5.,make_maps=True,female_file='data_female.xls',male_file='data_male.xls',
                 run_random_forest=True, rf_config=None, random_forest_only=False):
        self.root=Path(project_dir).expanduser().resolve()
        self.data_dir=self._path(data_dir or 'data')
        self.output_dir=self._path(output_dir or 'output/redesign/nine_variables_publication')
        if self.output_dir==self.data_dir or self.data_dir in self.output_dir.parents:
            raise ValueError('Output directory cannot be inside data directory')
        self.correlation_threshold,self.vif_threshold=correlation_threshold,vif_threshold
        self.make_maps=make_maps
        self.run_random_forest=run_random_forest
        self.rf_config=rf_config or RandomForestConfig()
        self.random_forest_only=random_forest_only
        if random_forest_only and not run_random_forest:
            raise ValueError("random_forest_only requires random forest to be enabled")
        self.female_file,self.male_file=female_file,male_file

    def _path(self,value):
        p=Path(value).expanduser()
        return (p if p.is_absolute() else self.root/p).resolve()

    def run(self):
        self.output_dir.mkdir(parents=True,exist_ok=True)
        state=dict(status='running',started_UTC=datetime.now(timezone.utc).isoformat())
        status_path=self.output_dir/'run_status.json'
        status_path.write_text(json.dumps(state,indent=2))
        try:
            print('[1/7] Descriptives and common initial variables...',flush=True)
            loader=MigrationDataLoader(self.data_dir,self.female_file,self.male_file)
            prepared=ModelData(loader,self.output_dir/'descriptive'/'model_setup').run()
            if self.random_forest_only:
                result=ComplementaryRandomForest(self.output_dir,self.rf_config).run(prepared)
                state.update(status='completed',mode='random_forest_only',
                             finished_UTC=datetime.now(timezone.utc).isoformat(),
                             comparison_folder=str(result['output_dir']))
                status_path.write_text(json.dumps(state,indent=2))
                print(f'Completed comparison: {result["output_dir"]}',flush=True)
                return self.output_dir
            MigrationDescriptives(loader,self.output_dir/'descriptive',self.data_dir/'kenya_counties.geojson').run(self.make_maps)
            print(f'      Same {len(prepared.sources)} source predictors / {len(prepared.features)} selected predictors for both sexes.',flush=True)
            print('[2/7] Separate female/male correlation screening...',flush=True)
            after_correlation={}
            for sex in ['female','male']:
                d=prepared.data[prepared.data.sex==sex].reset_index(drop=True)
                source=prepared.source_data[prepared.source_data.sex==sex].reset_index(drop=True)
                after_correlation[sex]=CorrelationAnalysis(self.output_dir/'correlation',self.correlation_threshold).run(d,prepared.features,sex,source)
            print('[3/7] Fractional logistic after correlation screening...',flush=True)
            summaries=[]
            first_effects={}
            for sex in ['female','male']:
                d=prepared.data[prepared.data.sex==sex].reset_index(drop=True)
                fit=LogisticAfterCorrelation(self.output_dir/'logistic_after_correlation').fit(d,after_correlation[sex],sex,prepared.dictionary)
                if fit is None: raise RuntimeError(f'{sex}: correlation-screened model not estimable')
                Plotter(self.output_dir/'logistic_after_correlation'/sex/'figures').model_results(
                    fit['effects'], pd.read_csv(self.output_dir/'logistic_after_correlation'/sex/'predictions.csv'), include_effects=False)
                first_effects[sex]=fit['effects']
                summaries.append(dict(analysis='after_correlation',**fit['diagnostics']))
            Plotter(self.output_dir/'logistic_after_correlation'/'figures').combined_marginal_effects(first_effects)
            print('[4/7] Separate iterative VIF screening...',flush=True)
            retained={}
            for sex in ['female','male']:
                d=prepared.data[prepared.data.sex==sex].reset_index(drop=True)
                retained[sex]=VIFAnalysis(self.output_dir/'vif',self.vif_threshold).run(d,after_correlation[sex],sex,initial_features=prepared.features,make_plots=False)
                print(f'      {sex}: {len(prepared.features)} initial -> {len(after_correlation[sex])} after correlation -> {len(retained[sex])} after VIF.',flush=True)
            vif_diagnostics=pd.concat([pd.read_csv(self.output_dir/'vif'/sex/'combined_vif_diagnostics.csv')
                                       for sex in ['female','male']],ignore_index=True)
            Plotter(self.output_dir/'vif'/'figures').combined_vif(vif_diagnostics,self.vif_threshold)
            print('[5/7] Fractional logistic after VIF screening...',flush=True)
            final_effects={}
            for sex in ['female','male']:
                d=prepared.data[prepared.data.sex==sex].reset_index(drop=True)
                fit=LogisticRetained(self.output_dir/'logistic_after_vif').fit(d,retained[sex],sex,prepared.dictionary)
                if fit is None: raise RuntimeError(f'{sex}: retained model not estimable')
                Plotter(self.output_dir/'logistic_after_vif'/sex/'figures').model_results(
                    fit['effects'], pd.read_csv(self.output_dir/'logistic_after_vif'/sex/'predictions.csv'), include_effects=False)
                final_effects[sex]=fit['effects']
                summaries.append(dict(analysis='after_vif',**fit['diagnostics']))
            Plotter(self.output_dir/'logistic_after_vif'/'figures').combined_marginal_effects(final_effects)
            print('[6/7] Full nine-variable fractional-logit sensitivity models...',flush=True)
            full_effects={}
            for sex in ['female','male']:
                d=prepared.data[prepared.data.sex==sex].reset_index(drop=True)
                fit=LogisticFull(self.output_dir/'logistic_full').fit(d,prepared.features,sex,prepared.dictionary)
                if fit is None: raise RuntimeError(f'{sex}: full nine-variable model not estimable')
                full_effects[sex]=fit['effects']
                summaries.append(dict(analysis='full_nine',**fit['diagnostics']))
                Plotter(self.output_dir/'logistic_full'/sex/'figures').model_results(
                    fit['effects'],pd.read_csv(self.output_dir/'logistic_full'/sex/'predictions.csv'),include_effects=False)
            Plotter(self.output_dir/'logistic_full'/'figures').combined_marginal_effects(full_effects)
            if self.run_random_forest:
                print('[7/7] Nested county comparison: forest versus full fractional logit...',flush=True)
                ComplementaryRandomForest(self.output_dir,self.rf_config).run(prepared)
            else:
                print('[7/7] Complementary prediction disabled.',flush=True)
            pd.DataFrame(summaries).to_csv(self.output_dir/'model_summary.csv',index=False)
            pd.DataFrame([dict(variable=f,female_retained=f in retained['female'],male_retained=f in retained['male'])
                          for f in prepared.features]).to_csv(self.output_dir/'retained_set_comparison.csv',index=False)
            state.update(status='completed',finished_UTC=datetime.now(timezone.utc).isoformat(),correlation_threshold=self.correlation_threshold,vif_threshold=self.vif_threshold,
                         candidate_source_columns=len(prepared.sources),encoded_initial_predictors=len(prepared.features),
                         retained=retained,initial_predictors=prepared.features,
                         random_forest_enabled=self.run_random_forest,
                         after_correlation=after_correlation, note='Exploratory screening; conditional model inference; education measurement flags require review')
            status_path.write_text(json.dumps(state,indent=2))
            print(f'Completed. Results: {self.output_dir}',flush=True)
            return self.output_dir
        except Exception as exc:
            state.update(status='failed',error=str(exc));status_path.write_text(json.dumps(state,indent=2))
            raise
