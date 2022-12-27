import numpy as np

class Individual:

    def __init__(self, route:list):
        self.route = np.array(route, dtype=int)
        self.size = len(self.route)
        self.cost = -1
        self.is_feasible = True

    def __repr__(self):
        return str(self.route)
