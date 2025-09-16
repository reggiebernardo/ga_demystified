import numpy as np
import pygad

class GeneticAlgorithm:
    def __init__(self, llprob, prior, p0, \
                 pop_size=2000, n_gen=100, sel_rate=0.3, crossover_prob=0.5, \
                 mutation_prob=[0.5, 0.3], parent_selection_type="rws", \
                 crossover_type="scattered", mutation_type="adaptive", num_threads=8):
        self.llprob = llprob
        self.prior = prior
        self.p0 = p0
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.sel_rate = sel_rate
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.parent_selection_type = parent_selection_type
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.num_threads = num_threads

        self.num_genes = len(prior)

    def fitness_func(self, ga_instance, chromosome, chromosome_idx):
        return np.exp(self.llprob(chromosome) - self.llprob(self.p0))

    def generate_arrays(self, prior, pop_size):
        init_uni = np.array([[np.random.uniform(p[0], p[1]) for p in prior] for _ in range(pop_size)])
        gene_space = [{'low': p[0], 'high': p[1]} for p in prior]
        return init_uni, gene_space

    def run_ga(self):
        init_uni, gene_space = self.generate_arrays(self.prior, self.pop_size)
        keep_parents = int(self.sel_rate * self.pop_size)

        ga_instance = pygad.GA(initial_population=init_uni, \
                               num_genes=self.num_genes, \
                               num_generations=self.n_gen, \
                               num_parents_mating=keep_parents, \
                               fitness_func=self.fitness_func, \
                               parent_selection_type=self.parent_selection_type, \
                               keep_parents=keep_parents, \
                               crossover_type=self.crossover_type, \
                               crossover_probability=self.crossover_prob, \
                               mutation_type=self.mutation_type, \
                               mutation_probability=self.mutation_prob, \
                               gene_space=gene_space, \
                               parallel_processing=self.num_threads)

        # Run the genetic algorithm
        ga_instance.run()
        return ga_instance
