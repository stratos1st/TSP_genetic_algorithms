from individual import Individual
import random
from numpy import random
import numpy as np
from copy import deepcopy
from population import Population

def order_crossover(p1:Individual, p2:Individual)->Individual:
    start_pos = random.randint(0, p1.size//2)
    end_pos = start_pos + random.randint(2, p1.size//2)
    child = np.empty([p1.size])

    child[start_pos:end_pos] = p1.route[start_pos:end_pos]
    visited = set(child[start_pos:end_pos])

    i = end_pos
    j = end_pos
    while j != start_pos:
        try:
            if p2.route[i] not in visited:
                child[j] = p2.route[i]
                visited.add(p2.route[i])
                j += 1
            i += 1
        except IndexError:
            j = j%p1.size
            i = i%p1.size

    return Individual(child)

def scramble_mutation(ind:Individual):
    start = random.randint(0,ind.size//2)
    end = start + random.randint(2,ind.size//2)
    np.random.shuffle(ind.route[start:end])

def k_tournament_elimination(pop:Population, k:int)->Individual:
    selected = np.random.choice(pop, size=k, replace=False)
    values = [indiv.cost for indiv in selected]
    return selected[np.argmax(values)]

def k_tournament_selection(pop:Population, k:int)->Individual:
    selected = np.random.choice(pop, size=k, replace=False)
    values = [individual.cost for individual in selected]
    return selected[np.argmin(values)]

def permutation_init(individual_size:int)->Individual:
    return Individual(list(np.random.permutation(individual_size)))

def random_init(individual_size:int, distance_matrix)->Individual:
    def get_possible_cities(curr_city):
        possible = []
        for destination in unvisited:
            if not np.isinf(distance_matrix[curr_city, destination]):
                possible.append(destination)
        return possible

    ranggee = range(individual_size)
    unvisited = set(ranggee)
    ans = [None]*individual_size
    ans[0] = random.randint(0,individual_size)
    curr_city = ans[0]
    for i in ranggee:
        possible_cities = get_possible_cities(curr_city)
        if len(possible_cities) == 0:
            curr_city = list(unvisited)[random.randint(0,len(unvisited))]
        else:
            curr_city = possible_cities[random.randint(0,len(possible_cities))]
        ans[i] = curr_city
        unvisited.remove(curr_city)

    return Individual(ans, list(ranggee))

def random_init2(individual_size:int, distance_matrix, max_tries=3)->Individual:
    ranggee = range(individual_size)
    unvisited = set(ranggee)
    ans = [None]*individual_size
    ans[0] = random.randint(0,individual_size)
    curr_city = ans[0]
    for i in ranggee:
        destination = list(unvisited)[random.randint(0,len(unvisited))]
        cnt = 0
        while np.isinf(distance_matrix[curr_city, destination]) and cnt!=max_tries:
            destination = list(unvisited)[random.randint(0,len(unvisited))]
            cnt +=1
        ans[i] = destination
        curr_city = destination
        unvisited.remove(curr_city)

    return Individual(ans, list(ranggee))

def opt2_local_search(ind:Individual, population:Population, max_steps=30):
    best = np.copy(ind.route)
    best_cost = population.__calculate_one_cost__(Individual(best))
    route = np.copy(ind.route)
    steps = 0
    start = random.randint(0,ind.size-3)

    for i in range(1+start, ind.size-2):
        if max_steps == steps: break
        for j in range(i+1, ind.size):
            if max_steps == steps: break
            if j-i == 1: continue # changes nothing, skip then
            new_route = np.copy(route)
            new_route[i:j] = route[j-1:i-1:-1] # this is the 2woptSwap
            if population.__calculate_one_cost__(Individual(new_route)) < best_cost:
                best = np.copy(new_route)
                best_cost = population.__calculate_one_cost__(Individual(best))
            steps += 1

    return Individual(best)

def swap_local_search(ind:Individual, population:Population, swaps=3):
    arr = [None] * swaps
    costs = [None] * swaps

    for i in range(swaps):
        new_ind = ind
        a = random.randint(0,new_ind.size)
        b = random.randint(0,new_ind.size)
        tmp = new_ind.get_city(b)
        new_ind.insert_city_at_possition(new_ind.get_city(a),b)
        new_ind.insert_city_at_possition(tmp,a)
        population.__calculate_one_cost__(new_ind)
        costs[i] = new_ind.cost
        #reverse changes
        tmp = new_ind.get_city(a)
        new_ind.insert_city_at_possition(new_ind.get_city(b),a)
        new_ind.insert_city_at_possition(tmp,b)

    a, b = arr[np.argmin(costs)]
    tmp = ind.get_city(b)
    ind.insert_city_at_possition(ind.get_city(a),b)
    ind.insert_city_at_possition(tmp,a)


# p1 = Individual([0,1,2,3,4,5,6,7,8])
# p2 = Individual([8,2,6,7,1,5,4,0,3])
# print(order_crossover(p1,p2))
