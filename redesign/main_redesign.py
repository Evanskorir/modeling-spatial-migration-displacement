"""Run this file in PyCharm or use python redesign/main_redesign.py."""
import argparse
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from redesign.runner import AnalysisRunner
    from redesign.random_forest_analysis import RandomForestConfig
else:
    from .runner import AnalysisRunner
    from .random_forest_analysis import RandomForestConfig


class Main:
    def __init__(self,argv=None):
        p=argparse.ArgumentParser(description='Nine-variable fractional logit and nested county random forest comparison')
        p.add_argument('--project-dir',type=Path,default=Path(__file__).resolve().parents[1])
        p.add_argument('--data-dir',type=Path,default=None)
        p.add_argument('--output-dir',type=Path,default=None)
        p.add_argument('--correlation-threshold',type=float,default=.8)
        p.add_argument('--vif-threshold',type=float,default=5.)
        p.add_argument('--female-file',default='data_female.xls')
        p.add_argument('--male-file',default='data_male.xls')
        p.add_argument('--no-maps',action='store_true')
        p.add_argument('--no-random-forest',action='store_true')
        p.add_argument('--random-forest-only',action='store_true',
                       help='Run complementary prediction without redoing descriptive/inference stages')
        p.add_argument('--rf-seed',type=int,default=20261125)
        p.add_argument('--rf-folds',type=int,default=5)
        p.add_argument('--rf-repeats',type=int,default=5)
        p.add_argument('--rf-inner-folds',type=int,default=3)
        p.add_argument('--rf-trees',type=int,default=300)
        p.add_argument('--rf-permutations',type=int,default=10)
        p.add_argument('--rf-jobs',type=int,default=2)
        self.args=p.parse_args(argv)

    def run(self):
        a=self.args
        return AnalysisRunner(a.project_dir,a.data_dir,a.output_dir,a.correlation_threshold,a.vif_threshold,
                              not a.no_maps,a.female_file,a.male_file,
                              run_random_forest=not a.no_random_forest,
                              rf_config=RandomForestConfig(seed=a.rf_seed,outer_splits=a.rf_folds,
                                  repeats=a.rf_repeats,inner_splits=a.rf_inner_folds,
                                  n_estimators=a.rf_trees,permutation_repeats=a.rf_permutations,
                                  n_jobs=a.rf_jobs),
                              random_forest_only=a.random_forest_only).run()


if __name__=='__main__':
    Main().run()
