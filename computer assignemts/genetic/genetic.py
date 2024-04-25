import random
from math import sin,cos , log10
import math



class DNA:
    def __init__(self, evaluation, solution):
        if solution is None:
            self.solution = [random.uniform(-10.0, 10.0) for _ in range(3)]
        else:
            self.solution = solution

        self.fitness = evaluation.eval_fitness(self.solution)


class Population:
    size = 500
    coefficient_range = 2
    mutation = []
    selection = []
    offsprings = []
    count_evolve = 0

    def __init__(self, fitness):
        self.fitness = fitness
        self.population = [DNA(fitness, None) for _ in range(Population.size)]
        self.sort()
        # while len(self.population) < Population.size:
        #     new_dna = DNA(fitness)
        #     if new_dna.fitness < 1000:
        #         self.population.append(new_dna)

    def sort(self):
        self.population = sorted(self.population, key=lambda dna: dna.fitness)

    # returns top 25 members
    def selection_process(self):
        self.selection = self.population[0:int(self.size / 2)]
        self.count_evolve += 1

    # generates 25 offsprings from top members
    def crossover_process(self):
        self.offsprings = []
        top = self.selection
        while len(self.offsprings) < int(Population.size / 2):
            parent1 = random.choice(top)
            parent2 = random.choice(top)
            if parent1.solution == parent2.solution:
                continue
            offspring = [(parent1.solution[0] + parent2.solution[0]) / 2,
                         (parent1.solution[1] + parent2.solution[1]) / 2,
                         (parent1.solution[2] + parent2.solution[2]) / 2]  # arithmetic mean for generating offsprings
            self.offsprings.append(DNA(self.fitness, offspring))


    def top_x_var(self,x):
        top_x = list(map(lambda x : x.fitness, self.population[0:x]))
        mean = sum(top_x)/x
        var = 0
        for i in range(x):
            var += math.pow(top_x[i] - mean,2)
        return var/x

    def mutation_process(self):
        self.mutation = []
        # for offspring in self.offsprings:
        #     point = offspring.solution
        #     i = random.randint(0,2)
        #     point[i] = random.uniform(-10,10)
        #     self.mutation.append(DNA(self.fitness,point))
        variance = self.top_x_var(20)
        # if variance < 1e-1:
        #     print(variance)
        # bias = 0.01 if variance < 1e-4 else 0.1 # TODO dynamic bias
        best_fitness = self.population[0].fitness
        if best_fitness >= 10:
            bias = 0.5
        elif best_fitness >= 1:
            bias = 0.1
        else:
            bias = random.uniform(-2,2)
        for sol in self.offsprings:
            point = sol.solution


            while True:
                mutated_cors = []

                for i in range(3):
                    cond = random.randint(1,3)
                    if cond == 1:
                        mutated_cors.append(point[i] + bias)
                    elif cond == 2:
                        mutated_cors.append(point[i] - bias)
                    else:
                        mutated_cors.append(point[i])

                if mutated_cors != point:  # make sure the mutated point is not the same
                    break
            self.mutation.append(DNA(self.fitness, mutated_cors))



    def new_population(self):
        temp = self.population + self.offsprings + self.mutation  # 100 + 50 + 50
        self.population = temp
        self.sort()
        self.population = self.population[0:self.size]


class Evaluation:

    def __init__(self, alpha, beta, teta):
        self.alpha = alpha
        self.beta = beta
        self.teta = teta

    def equation1(self, solution):
        x = solution[0]
        y = solution[1]
        z = solution[2]
        return (self.alpha * x) + (y * (x**2)) + (y ** 3) + (z ** 3)

    def equation2(self, solution):
        x = solution[0]
        y = solution[1]
        z = solution[2]
        return (self.beta * y) + sin(y) + (2** y) - z + log10(abs(x) + 1)

    def equation3(self, solution):
        x = solution[0]
        y = solution[1]
        z = solution[2]
        return (self.teta * z) + y - (cos(x + y) /
                                      (sin((z * y) - (y ** 2) + z) + 2))

    def eval_fitness(self, solution):
        return self.equation1(solution) ** 2 + self.equation2(solution) ** 2 + self.equation3(solution) ** 2


def solver(alpha, beta, teta):
    flag = True
    precision = 1e-8
    evolve_threshold = 50
    while flag:

        fitness = Evaluation(alpha, beta, teta)
        population = Population(fitness)
        generation = population.population

        while generation[0].fitness > precision:
            population.selection_process()
            population.crossover_process()
            population.mutation_process()
            population.new_population()
            generation = population.population
            if  population.count_evolve > evolve_threshold:
                # print("NEW POPULATION")
                break

        if generation[0].fitness < precision:
            flag = False


    x = generation[0].solution[0]
    y = generation[0].solution[1]
    z = generation[0].solution[2]
    return x, y, z


def main():
    print(solver(1, 1, 1))

if __name__ == "__main__":
    main()
