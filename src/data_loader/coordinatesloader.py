import os
import json


class CoordinateLoader:
    def __init__(self):
        project_path = os.path.dirname(os.path.realpath(__file__))
        self.coordinates_folder = os.path.join(project_path, "../../data/coordinates")
        self.coordinates = self._load_coordinates()

    def _load_coordinates(self):
        coords = {}
        for filename in os.listdir(self.coordinates_folder):
            if filename.endswith('.json'):
                # Normalize county name
                county_name = filename.replace('.json', '').replace('-', ' ').title()
                file_path = os.path.join(self.coordinates_folder, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    lon, lat = data['center']  # JSON uses [lon, lat]
                    coords[county_name] = (lat, lon)
        return coords

    def get_coordinates(self, county):
        county_key = county.strip().replace('-', ' ').title()
        return self.coordinates.get(county_key)

    def get_all_coordinates(self):
        return self.coordinates
