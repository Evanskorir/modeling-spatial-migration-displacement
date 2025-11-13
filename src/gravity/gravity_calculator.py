class GravityModel:
    def __init__(self, county_emigrant_share, population,
                 distance_calculator, distance_exponent=2.0):
        """
        county_emigrant_share: dict {county -> share}, share can be percent
        (e.g., 18.4) or fraction (e.g., 0.184)
        population:             dict {county -> population count}
        distance_calculator:    object with get_all_distances() -> {(origin, dest):
        distance_km}
        distance_exponent:      beta in D_ij^beta (default 2)
        """
        self.shares = county_emigrant_share
        self.population = population
        self.distance_calculator = distance_calculator
        self.beta = distance_exponent
        self.gravity_matrix = {}

    def _to_fraction(self, x):
        """Accept percent or fraction; return fraction in [0,1]."""
        return x / 100.0 if x is not None and x > 1 else x

    def emigrant_counts(self):
        """
        Convert share s_i to count M_i = s_i * P_i for each origin county.
        Returns: dict {county -> emigrant_count}
        """
        counts = {}
        for county, s in self.shares.items():
            if county not in self.population:
                continue
            frac = self._to_fraction(s)
            if frac is None:
                continue
            counts[county] = frac * self.population[county]
        return counts

    def compute_gravity(self):
        """
        G_ij = (M_i * P_j) / (D_ij ^ beta)
        where:
          M_i = emigrants from origin i (from share * origin population)
          P_j = destination population
          D_ij = distance between i and j
        """
        M = self.emigrant_counts()
        distances = self.distance_calculator.get_all_distances()

        for (origin, destination), D_ij in distances.items():
            if D_ij is None or D_ij == 0:
                continue
            if origin not in M or destination not in self.population:
                continue

            M_i = M[origin]
            P_j = self.population[destination]
            G_ij = (M_i * P_j) / (D_ij ** self.beta)
            self.gravity_matrix[(origin, destination)] = G_ij

    def get_gravity(self, origin, destination):
        return self.gravity_matrix.get((origin, destination))

    def get_all_gravity(self):
        return self.gravity_matrix


