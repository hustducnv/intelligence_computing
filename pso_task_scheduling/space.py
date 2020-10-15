import numpy as np
from config import Config


class Space(object):
    def __init__(self, n_particles, fitness_fn, w, c1, c2):
        self.n_particles = n_particles
        self.fitness_fn = fitness_fn
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.particles = []
        self.gbest_value = float('inf')
        self.gbest_position = None

    def evaluate_fitness(self):
        fitnesses = []
        for particle in self.particles:
            fitnesses.append(self.fitness_fn(particle))
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
            self.gbest_position = self.particles[best_fitness_candidate_index].position

    def move_particles(self):
        for particle in self.particles:
            new_velocity = \
                self.w * particle.velocity \
                + self.c1 * np.random.uniform() * (particle.pbest_position - particle.position) \
                + self.c2 * np.random.uniform() * (self.gbest_position - particle.position)
            # TODO: if new position is invalid? -> velocity?

            new_velocity = (new_velocity - new_velocity.min()) / (new_velocity.max() - new_velocity.min()) \
                           * Config.V_MAX*2 - Config.V_MAX
            particle.velocity = new_velocity
            particle.move()

    def search(self, n_iterations):
        iteration = 1
        while iteration <= n_iterations:
            self.update_pbest_gbest()
            print('iteration {}/{}: gbest_value = {}'.format(iteration, n_iterations, self.gbest_value))
            self.move_particles()
            iteration += 1

        print('the best solution is: {}'.format(self.gbest_position))
        print('best_value: {}'.format(self.gbest_value))
