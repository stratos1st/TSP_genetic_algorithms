import numpy as np
from individual import Individual
from itertools import pairwise

class Population:

    def __init__(self, population_size:int, distance_matrix, has_inf:bool=False):
        self.population_size = population_size
        self.distance_matrix = distance_matrix
        self.population = np.empty(self.population_size, dtype=Individual)
        self.has_inf = has_inf
        self.best_solution = None
        self.mean_objective = 0
        self.pairs = list(pairwise(range(0, distance_matrix.shape[0])))+[(distance_matrix.shape[0]-1,0)]

    def initialization(self, init_function):
        for i in range(self.population_size):
            self.population[i] = init_function(self.distance_matrix.shape[0])

    def selection_variation(self, number_of_offsprings:int, mutation_prob:float, selection_function, crossover_function, mutation_function):
        offspring = np.empty(number_of_offsprings, dtype=Individual)
        for i in range(number_of_offsprings):
            parent1 = selection_function(pop=self.population)
            parent2 = selection_function(pop=self.population)
            offspring[i] = crossover_function(p1=parent1, p2=parent2)
            if np.random.rand() <= mutation_prob:
                mutation_function(ind=offspring[i])

        self.population = np.concatenate((self.population, offspring))
        self.population_size = len(self.population)

    def local_search(self, opp_function, lso_prob:int):
        for i in self.population:
            if np.random.rand() <= lso_prob:
                i = opp_function(ind=i, population=self)

    def elimination(self, elimination_function, new_population_size):
        for i in range(self.population_size-new_population_size):
            self.population = self.population[self.population != elimination_function(pop=self.population)]
            self.population_size -= 1

    def calculate_stats(self):
        self.__calculate_costs__()
        costs = [ind.cost for ind in self.population]# if ind.is_feasible]
        self.best_solution = self.population[np.argmin(costs)]
        self.mean_objective = np.mean(costs)

    def __calculate_costs__(self):
        for indiv in self.population:
            if indiv.cost == -1:
                self.__calculate_one_cost__(indiv)

    def __calculate_one_cost__(self, indiv):
        value = 0
        for from_i, to_i in self.pairs:
            from_city = indiv.route[from_i]
            to_city = indiv.route[to_i]
            if self.has_inf:
                if np.isinf(self.distance_matrix[from_city, to_city]):
                    value += 200000
                    indiv.is_feasible = False
                else:
                    value += self.distance_matrix[from_city, to_city]
            else:
                value += self.distance_matrix[from_city, to_city]
        indiv.cost = value
        return value

    # def symetry(self):
    #     print(np.allclose(self.distance_matrix, self.distance_matrix.T, rtol=1e-05, atol=1e-08))
