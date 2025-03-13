import numpy as np
import random
import sys
from bee_class import Bee

MAX_ITER = 1000
max_distance = 2 # Maxium change to a vm that can be made.



"""
Each index is a task and the value at each index is the vm the task is assigned to.
bee = [1, 0, 2, 1, 2, 0]
"""

def find_neighbors(bee, num_vms, num_tasks):
    """
    Finds a neighbor of the current solution by randomly selecting a task and assigning it to a different vm.
    """
    neighbor = None
    distanceCount = 1

    while distanceCount <= max_distance:
        task = random.randint(0, num_tasks-1)
        neighbor1 = bee.vms.copy()
        neighbor2 = bee.vms.copy()
        # if the vm is already the smallest (1) then we wrap around to the largest vm
        if neighbor1[task] == 0:
            neighbor1[task] = num_vms - 1
        else:
            # otherwise decremnt the vm by 1
            neighbor1[task] = neighbor1[task] - 1

        # if the vm is already the largest (num_vms) then we wrap around to the smallest vm
        if neighbor2[task] == num_vms - 1:
            neighbor2[task] = 0
        else:
            neighbor2[task] == neighbor2[task] + 1

        fitness1 = f(neighbor1)
        fitness2 = f(neighbor2)
        if fitness1 > fitness2:
            neighborTemp, fitnessTemp = neighbor1, fitness1
        else:
            neighborTemp, fitnessTemp = neighbor2, fitness2
        if neighbor is None or fitness < fitnessTemp:
            neighbor = neighborTemp
            fitness =  fitnessTemp
        distanceCount += 1

    return neighbor



def generate_random_solution(num_tasks, num_vms):
    return np.random.randint(0, num_vms + 1, size=num_tasks)

def f(x):
    # The objective function.
    return np.random.randint(0, 100)

def solve(f, num_tasks, num_vms, num_bees=2, abandonment_limit=5):
    if num_bees < 2:
        print("Number of bees must be greater than 2")
        return -1

    #initialize the bees uniformly in the function space
    population = []

    for i in range(num_bees):
        new_bee = Bee(generate_random_solution(num_tasks, num_vms), f)
        population.append(new_bee)

    # fitness of population at initialization
    for bee in population:
        bee.update_fitness()

    local_best_idx = np.argmax([bee.fitness for bee in population])
    gloabal_best_vms = population[local_best_idx].vms
    best_fitness = population[local_best_idx].fitness

    # optimization
    for i in range(MAX_ITER):
        # employed bees
        for i, bee in enumerate(population):
            # get neighbor
            neighbor_sol = find_neighbors(bee, num_vms, num_tasks)
            neighbor_fitness = (f(neighbor_sol))

            # compare fitness with parent
            if neighbor_fitness > bee.fitness:
                bee.vms = neighbor_sol
                bee.fitness = neighbor_fitness
            else:
                bee.nonImprovementCounter += 1

        # calculate probabilities
        fitness_sum = np.sum([bee.fitness for bee in population])
        prob = [bee.fitness/fitness_sum for bee in population]

        # onlooker bees
        for i, bee in enumerate(population):
            if np.random.uniform() < prob[i]:
                # generate neighborhood source and test its fitness
                # generate a new food source similar to employed but slightly different. If the new food source fitness (rather than the neighborh fitness) is
                # higher, then we accept new solution, otherwise we increment nonImprovementCounter
                potential_neighbor = find_neighbors(bee, num_vms, num_tasks)

                # checks if the new neighbor is in the population
                def in_population(potential, population):
                    for bee in population:
                        if np.array_equal(potential, bee.vms):
                            return True
                    return False

                # if it isn't in the population create a new bee.
                if not in_population(potential_neighbor, population):
                    neighbor_bee = Bee(potential_neighbor, f)
                    neighbor_bee.update_fitness()
                    neighbor_fitness = neighbor_bee.fitness
                else:
                    neighbor_bee = None
                    neighbor_fitness = 0

                # recruit onlooker bees to richer sources of food
                if neighbor_bee is not None and neighbor_fitness > bee.fitness:
                    population[i] = neighbor_bee
                else:
                    bee.nonImprovementCounter += 1

        # scout bees
        for i, bee in enumerate(population):
            if bee.nonImprovementCounter >= abandonment_limit:
                # Generate a new food source randomly
                #TODO: Maybe implement something that tracks bad food sources and filters them out of
                # the random generation.
                scout_bee = Bee(generate_random_solution(num_tasks, num_vms), f)
                population[i] = scout_bee
                population[i].update_fitness()


        # update best solutions
        local_best_idx = np.argmax([bee.fitness for bee in population])
        if population[local_best_idx].fitness > best_fitness:
            global_best_vms = population[local_best_idx].vms
            best_fitness = population[local_best_idx].fitness

    return global_best_vms, best_fitness

def main():
    best_vms, best_fitness = solve(f, 20, 10)
    print(f"Best VMS: {best_vms}")
    print(f"Best Fitness: {best_fitness}")




if __name__ == "__main__":
    main()