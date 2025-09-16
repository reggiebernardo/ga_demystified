import os, sys
import numpy as np
from numpy.linalg import inv
from scipy.linalg import block_diag
from astropy.cosmology import LambdaCDM
from numpy.random import uniform as U
import pygad # GA python library

print('cpu count =', os.cpu_count())

if __name__ == '__main__':

    ########## 1 DATA ##########
    
    # import the data points
    filename = 'https://gitlab.com/mmoresco/CCcovariance/-/raw/master/data/HzTable_MM_BC03.dat'
    z_cc, Hz_cc, errHz_cc = np.genfromtxt(filename, comments = '#', usecols = (0,1,2), \
                                          unpack = True, delimiter = ',')
    
    # import and calculate covariance matrix
    # import CC covariance matrix data
    
    filename = 'https://gitlab.com/mmoresco/CCcovariance/-/raw/master/data/data_MM20.dat'
    zmod, imf, slib, sps, spsooo = np.genfromtxt(filename, comments = '#', \
                                                 usecols = (0,1,2,3,4), unpack = True)
    
    # calculate CC covariance matrix
    cov_mat_diag = np.zeros((len(z_cc), len(z_cc)), dtype = 'float64')
    
    for i in range(len(z_cc)):
        cov_mat_diag[i,i] = errHz_cc[i]**2
    
    imf_intp = np.interp(z_cc, zmod, imf)/100
    slib_intp = np.interp(z_cc, zmod, slib)/100
    sps_intp = np.interp(z_cc, zmod, sps)/100
    spsooo_intp = np.interp(z_cc, zmod, spsooo)/100
    
    cov_mat_imf = np.zeros((len(z_cc), len(z_cc)), dtype = 'float64')
    cov_mat_slib = np.zeros((len(z_cc), len(z_cc)), dtype = 'float64')
    cov_mat_sps = np.zeros((len(z_cc), len(z_cc)), dtype = 'float64')
    cov_mat_spsooo = np.zeros((len(z_cc), len(z_cc)), dtype = 'float64')
    
    for i in range(len(z_cc)):
        for j in range(len(z_cc)):
            cov_mat_imf[i,j] = Hz_cc[i] * imf_intp[i] * Hz_cc[j] * imf_intp[j]
            cov_mat_slib[i,j] = Hz_cc[i] * slib_intp[i] * Hz_cc[j] * slib_intp[j]
            cov_mat_sps[i,j] = Hz_cc[i] * sps_intp[i] * Hz_cc[j] * sps_intp[j]
            cov_mat_spsooo[i,j] = Hz_cc[i] * spsooo_intp[i] * Hz_cc[j] * spsooo_intp[j]
    
    C_cc = cov_mat_spsooo + cov_mat_imf + cov_mat_diag # full CC covariance
    C_cc_inv = inv(C_cc)
    
    
    # import supernovae data
    loc_lcparam = 'https://github.com/PantheonPlusSH0ES/DataRelease/raw/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat'
    
    # pantheon plus systematics
    lcparam = np.loadtxt(loc_lcparam, skiprows = 1, usecols = (2, 8, 9, 10, 11))
    
    # setup pantheon samples with z > 0.01
    z_pp = lcparam[:, 0][111:]
    # mz_pp = lcparam[:, 1][111:]
    # sigmz_pp = lcparam[:, 2][111:]
    mMz_pp = lcparam[:, 3][111:]
    sigmMz_pp = lcparam[:, 4][111:]
    
    # load the pantheon+ covariance matrix
    
    loc_lcparam_sys = 'https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov'
    
    lcparam_sys = np.loadtxt(loc_lcparam_sys, skiprows = 1)
    
    # pantheon plus systematics
    C_sne = lcparam_sys.reshape(1701, 1701)
    C_sne_inv = np.linalg.inv(C_sne[111:, 111:])
    
    
    ########## 2 FITNESS FUNCTION/likelihood ##########
    
    def Chi2_CC(x):
        # H0, Om0, Ok0 = x
        # cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=1-Om0-Ok0)
        H0, Om0, Ode0 = x
        cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
        dev_cc = Hz_cc - cosmo.H(z_cc).value
        return dev_cc.T @ C_cc_inv @ dev_cc
    
    def Chi2_SNe(x):
        # H0, Om0, Ok0 = x
        # cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=1-Om0-Ok0)
        H0, Om0, Ode0 = x
        cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
        dev_sne = mMz_pp - cosmo.distmod(z_pp).value
        return dev_sne.T @ C_sne_inv @ dev_sne
    
    
    # add flat priors
    H0_min, H0_max = 0, 100
    Om0_min, Om0_max = 0, 1
    # Ok0_min, Ok0_max = -0.5, 0.5
    Ode0_min, Ode0_max = 0, 1
    
    def llflatprior(x):
        H0, Om0, Ode0 = x
        # if (H0_min < H0 < H0_max and Om0_min < Om0 < Om0_max \
        #     and Ok0_min < Ok0 < Ok0_max):
        if (H0_min < H0 < H0_max and Om0_min < Om0 < Om0_max \
            and Ode0_min < Ode0 < Ode0_max):
            return 0.0
        return -np.inf
    
    def llprob_CC(x):
        H0, Om0, Ode0 = x
        if (Om0 < 0) or (H0 < 0):
            return -np.inf
        lp = llflatprior(x)
        lk = -0.5*Chi2_CC(x)
        if np.isnan(lk):
            return -np.inf
        return lp + lk if np.isfinite(lp) else -np.inf
    
    def llprob_CCSNe(x):
        H0, Om0, Ode0 = x
        if (Om0 < 0) or (H0 < 0):
            return -np.inf
        lp = llflatprior(x)
        lk = -0.5*(Chi2_CC(x) + Chi2_SNe(x))
        if np.isnan(lk):
            return -np.inf
        return lp + lk if np.isfinite(lp) else -np.inf
    
    
    def fitness_func(ga_instance, chromosome, chromosome_idx):
        return np.exp(llprob_CC(chromosome))
    
    
    # setup the population size and the initial population
    pop_size = 3000
    init_uni = np.array([[U(H0_min, H0_max), U(Om0_min, Om0_max), U(Ode0_min, Ode0_max)] \
                         for i in np.arange(pop_size)])
    
    gene_space = [{'low': H0_min, 'high': H0_max}, \
                  {'low': Om0_min, 'high': Om0_max}, {'low': Ode0_min, 'high': Ode0_max}]
    
    
    ########## 2 GA run ##########
    
    num_genes = 3 # length of chromosome
    
    n_gen = 100 # number of generations
    sel_rate = 0.3 # selection rate
    
    # parent selection
    parent_selection_type = "rws" # roulette wheel selection
    keep_parents = int(sel_rate*pop_size)
    num_parents_mating = int(sel_rate*pop_size)
    
    # crossover options: single_point, two_points, uniform
    crossover_type = "scattered"
    crossover_prob = 0.5
    
    # mutation type options: random, swap, inversion, scramble, adaptive
    mutation_type = "adaptive"
    mutation_prob_0 = [0.5, 0.3] # if adaptive, two numbers as [a, b]
    mutation_prob_1 = [0.8, 0.2] # if adaptive, two numbers as [a, b]
    
    def on_gen(ga_instance): # print results every N = 10 generations
        generation_count = ga_instance.generations_completed
        if generation_count % 10 == 0:
            print("Generation:", generation_count)
            # print("Fitness of the best solution:", ga_instance.best_solution()[1])
    
    # setup GA instance, for random initial pop.
    ga_instance = pygad.GA(initial_population = init_uni,
                           num_genes = num_genes,
                           num_generations = n_gen,
                           num_parents_mating = num_parents_mating,
                           fitness_func = fitness_func,
                           parent_selection_type = parent_selection_type,
                           keep_parents = keep_parents,
                           crossover_type = crossover_type,
                           crossover_probability = crossover_prob,
                           mutation_type = mutation_type,
                           mutation_probability = mutation_prob_0,
                           on_generation = on_gen,
                           gene_space = gene_space,
                           parallel_processing = os.cpu_count())
                           # save_best_solutions = True # to save solutions per gen.
    
    # perform GA run
    ga_instance.run()
    ga_instance.save(filename='ga_cc_mutation_0')
    
    ######## mutation 1 #########
    
    ga_instance = pygad.GA(initial_population = init_uni,
                           num_genes = num_genes,
                           num_generations = n_gen,
                           num_parents_mating = num_parents_mating,
                           fitness_func = fitness_func,
                           parent_selection_type = parent_selection_type,
                           keep_parents = keep_parents,
                           crossover_type = crossover_type,
                           crossover_probability = crossover_prob,
                           mutation_type = mutation_type,
                           mutation_probability = mutation_prob_1,
                           on_generation = on_gen,
                           gene_space = gene_space,
                           parallel_processing = os.cpu_count())
                           # save_best_solutions = True # to save solutions per gen.
    
    # perform GA run
    ga_instance.run()
    ga_instance.save(filename='ga_cc_mutation_1')
    
    
    ########## 2 GA run with CC + SNe ##########
    
    def fitness_func(ga_instance, chromosome, chromosome_idx):
        return np.exp(llprob_CCSNe(chromosome))
    
    
    # setup the population size and the initial population
    pop_size = 3000
    init_uni = np.array([[U(H0_min, H0_max), U(Om0_min, Om0_max), U(Ode0_min, Ode0_max)] \
                         for i in np.arange(pop_size)])
    
    gene_space = [{'low': H0_min, 'high': H0_max}, \
                  {'low': Om0_min, 'high': Om0_max}, {'low': Ode0_min, 'high': Ode0_max}]
    
    
    ########## 2 GA run ##########
    
    num_genes = 3 # length of chromosome
    
    n_gen = 100 # number of generations
    sel_rate = 0.3 # selection rate
    
    # parent selection
    parent_selection_type = "rws" # roulette wheel selection
    keep_parents = int(sel_rate*pop_size)
    num_parents_mating = int(sel_rate*pop_size)
    
    # crossover options: single_point, two_points, uniform
    crossover_type = "scattered"
    crossover_prob = 0.5
    
    # mutation type options: random, swap, inversion, scramble, adaptive
    mutation_type = "adaptive"
    mutation_prob_0 = [0.5, 0.3] # if adaptive, two numbers as [a, b]
    mutation_prob_1 = [0.8, 0.2] # if adaptive, two numbers as [a, b]
    
    def on_gen(ga_instance): # print results every N = 10 generations
        generation_count = ga_instance.generations_completed
        if generation_count % 10 == 0:
            print("Generation:", generation_count)
            # print("Fitness of the best solution:", ga_instance.best_solution()[1])
    
    # setup GA instance, for random initial pop.
    ga_instance = pygad.GA(initial_population = init_uni,
                           num_genes = num_genes,
                           num_generations = n_gen,
                           num_parents_mating = num_parents_mating,
                           fitness_func = fitness_func,
                           parent_selection_type = parent_selection_type,
                           keep_parents = keep_parents,
                           crossover_type = crossover_type,
                           crossover_probability = crossover_prob,
                           mutation_type = mutation_type,
                           mutation_probability = mutation_prob_0,
                           on_generation = on_gen,
                           gene_space = gene_space,
                           parallel_processing = os.cpu_count())
                           # save_best_solutions = True # to save solutions per gen.
    
    # perform GA run
    ga_instance.run()
    ga_instance.save(filename='ga_ccsne_mutation_0')
    
    ######## mutation 1 #########
    
    ga_instance = pygad.GA(initial_population = init_uni,
                           num_genes = num_genes,
                           num_generations = n_gen,
                           num_parents_mating = num_parents_mating,
                           fitness_func = fitness_func,
                           parent_selection_type = parent_selection_type,
                           keep_parents = keep_parents,
                           crossover_type = crossover_type,
                           crossover_probability = crossover_prob,
                           mutation_type = mutation_type,
                           mutation_probability = mutation_prob_1,
                           on_generation = on_gen,
                           gene_space = gene_space,
                           parallel_processing = os.cpu_count())
                           # save_best_solutions = True # to save solutions per gen.
    
    # perform GA run
    ga_instance.run()
    ga_instance.save(filename='ga_ccsne_mutation_1')
    
    

