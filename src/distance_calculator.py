import numpy as np

from geopy.distance import geodesic
from sklearn.metrics.pairwise import haversine_distances


class DistanceCalculator:
    def __init__(self, coordinates_dict, method="haversine"):
        self.coordinates_dict = coordinates_dict
        self.method = method.lower()
        self.distance_matrix = {}
        self.counties = list(coordinates_dict.keys())

    def compute_all_distances(self):
        if self.method == "haversine":
            self._compute_haversine_distances()
        elif self.method == "geodesic":
            self._compute_geodesic_distances()
        else:
            raise ValueError("Unsupported distance method. Use 'haversine' or 'geodesic'.")

    def _compute_haversine_distances(self):
        coord_list = [self.coordinates_dict[county] for county in self.counties]
        coord_array = np.radians(np.array(coord_list))
        radian_matrix = haversine_distances(coord_array)
        R = 6371
        km_matrix = radian_matrix * R

        for i, origin in enumerate(self.counties):
            for j, dest in enumerate(self.counties):
                if origin != dest:
                    self.distance_matrix[(origin, dest)] = km_matrix[i][j]

    def _compute_geodesic_distances(self):
        for origin in self.counties:
            for dest in self.counties:
                if origin != dest:
                    coord1 = self.coordinates_dict[origin]
                    coord2 = self.coordinates_dict[dest]
                    self.distance_matrix[(origin, dest)] = geodesic(coord1,
                                                                    coord2).kilometers

    def get_distance(self, county1, county2):
        return self.distance_matrix.get((county1, county2))

    def get_all_distances(self):
        return self.distance_matrix

    def get_distances_from_hub(self, hub_name="Nairobi"):
        """
        Returns a dict of distances from a selected hub county to every other county.
        """
        if hub_name not in self.counties:
            raise ValueError(f"{hub_name} not found in county list.")

        return {
            county: self.distance_matrix.get((hub_name, county))
            for county in self.counties if county != hub_name
        }


