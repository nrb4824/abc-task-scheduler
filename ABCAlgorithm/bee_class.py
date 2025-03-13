import numpy as np


class Bee():
    def __init__(self, vms, f) -> None:
        self.vms = vms
        self.objective_func = f
        self.fitness = 0
        self.nonImprovementCounter = 0

    def update_fitness(self):
        self.fitness = self.objective_func(self.vms)



