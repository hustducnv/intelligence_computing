import numpy as np


class Space(object):
    def __init__(self, n_particles, w, c1, c2):
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.particles = []
        self.gbest_value = float('inf')
        self.gbest_position = None

    # TODO : implement fitness
    def fitness(self, particle):
        return 1

    def evaluate_fitness(self):
        fitnesses = []
        for particle in self.particles:
            fitnesses.append(fitnesses(particle))
        return fitnesses

    def update_pbest_gbest(self):
        fitnesses = self.evaluate_fitness()

        # update pbest
        for idx, particle in enumerate(self.particles):
            fitness_candidate = fitnesses[idx]
            if fitness_candidate < particle.pbest_value:
                particle.pbest_value = fitness_candidate
                particle.pbest_position = particle.position

        # update gbest
        best_fitness_candidate_index = np.argmin(fitnesses)
        if fitnesses[best_fitness_candidate_index] < self.gbest_value:
            self.gbest_value = fitnesses[best_fitness_candidate_index]
            self.gbest_position = self.particles[best_fitness_candidate_index]

    def move_particles(self):
        for particle in self.particles:
            new_velocity = \
                self.w * particle.velocity \
                + self.c1 * np.random.uniform() * (particle.pbest_position - particle.position) \
                + self.c2 * np.random.uniform() * (self.gbest_position - particle.position)
            particle.velocity = new_velocity
            particle.move()
