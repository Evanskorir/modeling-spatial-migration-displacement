"""Nine common, explicit candidates; source values are never changed."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data_loader import OUTCOME

from .variables import CANDIDATES

class ModelData:
    def __init__(self, loader, output_dir):
        self.loader, self.output_dir = loader, Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        raw = self.loader.get_long_data().sort_values(['sex', 'county_key']).reset_index(drop=True)
        # The input uses "Average Household Size"; fail clearly for other schemas.
        missing = [source for source, *_ in CANDIDATES if source not in raw]
        if missing:
            raise ValueError(f'Missing selected source columns: {missing}. Available: {list(raw.columns)}')
        self.sources = [c[0] for c in CANDIDATES]
        self.features = [c[1] for c in CANDIDATES]
        self.data = raw[['County', 'county_key', 'sex']].copy()
        self.data['y'] = raw[OUTCOME] / 100.
        self.source_data = self.data.copy()
        rows = []
        for priority, (source, key, method, domain) in enumerate(CANDIDATES, 1):
            values = raw[source].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f'Missing/nonfinite selected predictor: {source}; no counties dropped')
            if method == 'natural_log':
                if (values <= 0).any(): raise ValueError(f'{source} must be positive')
                values = np.log(values)
            elif method == 'percent_divided_by_100':
                if ((values < 0) | (values > 100)).any(): raise ValueError(f'Invalid percentage: {source}')
                values = values / 100.
            elif (values < 0).any() or (key == 'household_size' and (values <= 0).any()):
                raise ValueError(f'Invalid nonnegative source values: {source}')
            for sex in ['female', 'male']:
                if np.std(values[raw.sex.eq(sex)], ddof=0) <= 1e-12:
                    raise ValueError(f'{sex}: constant predictor {source}')
            mean, sd = values.mean(), values.std(ddof=0)
            self.source_data[key] = values
            self.data[key] = (values-mean)/sd
            rows.append(dict(source_column=source, feature=key, transformation=method,
                             reference_mean=mean, reference_SD=sd, priority=priority, domain=domain,
                             metadata_status='Source definition, reference year and denominator require verification',
                             education_caution=('Single source category, NOT secondary-or-higher; education totals have quality flags'
                                                if key == 'completed_secondary' else '')))
        self.dictionary = pd.DataFrame(rows)
        self.dictionary.to_csv(self.output_dir/'common_candidate_dictionary.csv', index=False)
        self.data.to_csv(self.output_dir/'prepared_model_data.csv', index=False)
        return self
