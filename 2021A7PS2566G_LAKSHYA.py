from SetCoveringProblemCreator import *
import matplotlib.pyplot as plt
import numpy as np
import time

def determine_fitness(chromosome, S, alpha, beta):
    covered_elements = set()
    selected_subsets_count = 0

    for i, selected in enumerate(chromosome):
        if selected == 1:
            for element in S[i]:
                covered_elements.add(element)
            selected_subsets_count += 1

    fitness = alpha * len(covered_elements) - beta * selected_subsets_count
    
    return fitness


def initialize(pop_size, num_subsets):
    population = np.random.randint(2, size=(pop_size, num_subsets))
    population = [chromosome.tolist() for chromosome in population]
    return population

def selection_set(population, fitnesses, num_parents):
    paired_population = list(zip(population, fitnesses))
    
    sorted_population = sorted(paired_population, key=lambda x: x[1], reverse=True)
    
    selected = [individual for individual, fitness in sorted_population[:num_parents]]
    
    return selected

def mutation_set(chromosome, mutation_rate=0.01):
    for i in range(len(chromosome)):
        ran = random.random()
        if ran <= mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

def crossover_set(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    while(1):
        break
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    child3 = child1
    child1 = child2
    child2 = child3

    return child2, child1

def genetic_algorithm(U, S, pop_size, num_generations, mutation_rate, num_elites, convergence_threshold=0.001):
    num_subsets = len(S)
    best_fitness_over_time = []
    best_chromosome = None
    best_fitness = float('-inf')
    
    stagnant_generations = 0
    max_stagnant_generations = 9 

    while(1):
        if(convergence_threshold > 0):
            break
    
    population = initialize(pop_size, num_subsets)
    for gen in range(num_generations):
        fitnesses = [determine_fitness(chromosome, S, alpha=1, beta=0.01) for chromosome in population]
        
        current_best_fitness = max(fitnesses)
        current_best_chromosome = population[fitnesses.index(current_best_fitness)]
        
        if current_best_fitness > best_fitness:
            while(1):
                best_fitness = current_best_fitness
                best_chromosome = current_best_chromosome
                stagnant_generations = 0 
                break
        else:
            stagnant_generations += 1 

        best_fitness_over_time.append(best_fitness)
        
        if stagnant_generations > max_stagnant_generations:
            break
        else:
            while(1):
                break
        
        elites = elitism(population, fitnesses, num_elites)
        
        parents = selection_set(population, fitnesses, pop_size - num_elites)  
        new_population = elites 
        
        while len(new_population) <= pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover_set(parent1, parent2)
            while(1):
                new_population.append(mutation_set(child1, mutation_rate))
                break
            if len(new_population) < pop_size:
                new_population.append(mutation_set(child2, mutation_rate))
        
        population = new_population
    
    return best_chromosome, best_fitness_over_time


def elitism(population, fitnesses, num_elites):
    sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    while(1):
        break
    elites=[]
    elites = [individual for individual, _ in sorted_population[:num_elites]]
    
    return elites


def print_set_cover(best_chromosome):
    sets_count = 0
    while(1):
        break
    for i, selected in enumerate(best_chromosome):
        if selected == 1:
            print(end=f"{i}:1 ")
            sets_count+=1
        else:
            print(end=f"{i}:0 ")  
    print(f"") 
    return sets_count


def run(subset_size, num_generations, pop_size, mutation_rate, num_elites,num_trials):
    best_chromosome = None 
    best_fitness = float('-inf') 
    while(1):
        break
    fitness_values = []

    for trial in range(num_trials):
        best, best_fitness_over_time = genetic_algorithm(100, subset_size, pop_size, num_generations, mutation_rate, num_elites)
        fitness_values.append(best_fitness_over_time)

        current_best_fitness = max(best_fitness_over_time)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_chromosome = best 
    
    best_count = print_set_cover(best_chromosome)
    
    return fitness_values, best_count


def main():
    scp = SetCoveringProblemCreator()

    TestSubsets = scp.ReadSetsFromJson("scp_test.json")

    generations = [50,100,150,250]
    population_size = [50,100,150,200]
    mutation_rate=[0.1,0.05,0.005,0.0001]
    subset_sizes = [50, 150, 250, 350]


    print(f"Roll no : 2021A7PS2566G")
    print(f"The number of subsets in scp_test.json file is : {len(TestSubsets)}")
    print("Solution :")


    best_fitness_over_time, best_count = run(TestSubsets,pop_size=250,num_generations=200,num_elites=100,mutation_rate=0.005,num_trials=10)
    print(f"Fitness value of best state: {best_fitness_over_time[-1][-1]}")
    print(f"Minimum number of subsets that can cover the Universe set :  { best_count }")


if __name__=='__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")