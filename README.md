# Gendered Patterns of Internal Migration in Kenya: A Gravity Model Approach

## Introduction
This repository implements a Python-based empirical framework for analyzing gender-differentiated internal migration patterns across Kenyan counties using a gravity model approach. Migration flows are modeled as an outward spatial interaction process originating from Mandera County, a structurally disadvantaged region characterized by persistent net out-migration.
The project estimates both baseline and extended gravity models using county-level demographic, economic, and geographic data drawn primarily from the 2022 Kenya Demographic and Health Survey (KDHS) and supplementary administrative sources. Models are estimated separately for women and men to identify gender-specific migration determinants and structural asymmetries in internal mobility.


## Data Files
```
data: Socio-economic indicators and cumulative cases collected from different sources "
   ├── coordinates      
   │   ├── coordinates   # JSON files containing latitude and longitude of a county
   ├── data_male.xls   # Excel file with male data such as population, GDP, poverty, working population, etc.
   ├── data_female.xls # Excel file with female data relating to the migration      
   └── kenya counties geojson  # JSON files containinsuch as populati g latitude and longitude 
```

## Folder Structure
```
data                
src                    
 ├── data_loader        
 │   ├── coordinatesloader     
 │   └── dataloader   
 ├── gravity            
 │   ├── gravity_base_model 
 │   ├── gravity_calculator    
 │   └── gravity_migration_model   
 ├── clustering
 ├── distance_calculator
 ├── eba
 ├── plotter               
 └── runner
main 
README
```
## File Details
#### `src/data_loader/`
- **`coordinatesloader.py`**: Loads and normalizes geographic coordinates (latitude and longitude) for Kenyan counties 
from JSON files to support spatial analysis and mapping.
- **`dataloader.py`**: Loads, preprocesses, and provides access to migration, demographic, 
socioeconomic, and geospatial data for use in modeling and visualization.
#### `src/gravity/`
- **`gravity_base_model.py`**: Provides a framework for gravity-based regression modeling, including feature extraction, 
data preparation, and automated variable selection using backward elimination for spatial epidemiological analysis.
- **`gravity_calculator.py`**: Computes a gravity matrix estimating interaction strength between counties based on 
socioeconomic data, migration, population, and distances.
- **`gravity_migration_model.py`**:  A gravity-based model for analyzing migration using migration data, distance, 
population,  and socioeconomic indicators.
#### `src/`
- **`clustering.py`**:  Performs hierarchical clustering of counties for the genders based on selected significant 
socioeconomic indicators.
- **`distance_calculator.py`**: Computes pairwise distances between counties using either geodesic or 
haversine methods for spatial analysis.
- **`plotter.py`**: Generates a wide range of visualizations—including heatmaps, scatterplots, and dendrograms to 
analyze spatial migration dynamics across Kenyan counties.
- **`runner.py`**: Coordinates the entire analysis pipeline—data loading, gravity modeling, spatial regression, 
visualization, and clustering—for migration patterns across Kenyan counties.
- **`main.py`**: Runs the full migration spatial analysis pipeline by initializing and executing the Runner

## Implementation
To run the spatial analysis, follow these steps:
1. Open `main.py` 
2. Run the analysis with these steps:` 
#### Initialize the AnalysisOrchestrator
```  analysis = Runner() ```
#### generate all the necessary plots
```run.run() ```

## Output
```
output/clusters
     ├── clusters_female_count.csv
     ├──  clusters_female_percent.csv
     ├──  clusters_male_count.csv
     ├──  clusters_male_percent.csv
 output/eba
     ├──  gender
 output  
     ├──  clusters_male_percent.csv
     ├──  ols_full_gender_county.pdf 
     ├──  correlation_matrix.pdf
     ├──   county_stacked_percent.pdf    
     ├──  migration_overlay_county.png
```

## Requirement
This project is developed and tested with Python 3.8 or higher. Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt

