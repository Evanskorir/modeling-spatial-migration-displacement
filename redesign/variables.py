"""Single source of truth for the nine candidates and redundancy priority."""
# Order is a prespecified redundancy priority, not an importance ranking.
CANDIDATES = [
    ('Population Density', 'log_population_density', 'natural_log', 'Settlement concentration'),
    ('Poverty Rate', 'poverty_rate', 'percent_divided_by_100', 'Economic disadvantage'),
    ('Currently_employed', 'currently_employed', 'percent_divided_by_100', 'Employment'),
    ('Completed_secondary', 'completed_secondary', 'percent_divided_by_100', 'Education: completed-secondary category ONLY'),
    ('Average Household Size', 'household_size', 'identity', 'Household structure'),
    ('Mass_Media_Access', 'mass_media_access', 'percent_divided_by_100', 'Information access'),
    ('Daily_Internet_Access', 'daily_internet_access', 'percent_divided_by_100', 'Digital connectivity'),
    ('Gini_coefficient', 'gini_coefficient', 'identity', 'Income inequality; original source scale'),
    ('Acceptable_Food_Consumption_Score', 'acceptable_food_consumption', 'percent_divided_by_100', 'Food security'),
]


FEATURE_LABELS = {key: source.replace("_", " ") for source, key, _, _ in CANDIDATES}

FEATURE_LABELS['log_population_density'] = 'Log population density'
