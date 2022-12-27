import Reporter
import numpy as np
from individual import Individual
from population import Population
import funcs
from functools import partial
import sys, os

# cls & python -m cProfile -o out.prof r0123456.py
# snakeviz out.prof

file_name = "./data/tour750.csv"
has_inf = False
show_feasible = False
progress_every = 10

initial_population_size = 1000
population_size = 500
offspring_size = 500
k = 3
mutation_prob = 0.01
lso_prob = 0

# Modify the class name to match your student number.
class r0123456:

	def __init__(self, filename):
		self.reporter = Reporter.Reporter(filename)

	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()



		# initialization_function = partial(funcs.random_init2,distance_matrix=distanceMatrix)
		# initialization_function = partial(funcs.random_init,distance_matrix=distanceMatrix)
		initialization_function = funcs.permutation_init
		selection_function = partial(funcs.k_tournament_selection, k=k)
		crossover_function = funcs.order_crossover
		mutation_function = funcs.scramble_mutation
		lsopp_funtion = partial(funcs.opt2_local_search, max_steps=300)
		elimination_function = partial(funcs.k_tournament_elimination, k=k)





		# initialization
		print("Initialization...")
		population = Population(population_size=initial_population_size, distance_matrix=distanceMatrix, has_inf=has_inf)
		population.initialization(init_function=initialization_function)
		population.local_search(opp_function=lsopp_funtion, lso_prob=1)
		lsopp_funtion = funcs.opt2_local_search
		population.calculate_stats()
		if show_feasible:
		    self.__print_feasible__(population)
		print("Initialization done")

		mean_objectives = [population.mean_objective]
		rsd_mean = 1
		iterations = 0
		# while rsd_mean>0.005 or iterations<20:
		while True:
			population.selection_variation(number_of_offsprings=offspring_size, mutation_prob=mutation_prob,
											selection_function=selection_function, crossover_function=crossover_function,
											mutation_function=mutation_function)
			population.local_search(opp_function=lsopp_funtion, lso_prob=lso_prob)
			population.calculate_stats()
			population.elimination(elimination_function=elimination_function, new_population_size=population_size)

			# prints
			iterations += 1
			mean_objectives.append(population.mean_objective)
			rsd_mean = np.std(mean_objectives[-15:])/np.mean(mean_objectives[-15:])
			if not iterations%progress_every:
				print(iterations)
				if show_feasible:
					self.__print_feasible__(population)

			#time limit
			timeLeft = self.reporter.report(population.mean_objective, population.best_solution.cost, population.best_solution.route)
			if timeLeft < 0:
				break

		print("timeLeft= "+str(timeLeft))
		return 0

	def __print_feasible__(self, population):
		cnt = 0
		for i in population.population:
			if not i.is_feasible:
				cnt += 1
		print("Solutions not feasible= "+str(cnt))
		print(population.best_solution.is_feasible)


if __name__ == "__main__":
	# sys.stdout = open(os.devnull, 'w')
	filename = "test"
	object = r0123456(filename)
	object.optimize(file_name)
	# sys.stdout = sys.__stdout__
