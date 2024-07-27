
# -------- Libs --------

# For es
import torch # the main pytorch library
from torch import nn # the sub-library containing neural networks
from torch.utils.data import DataLoader, Subset # an object that generates batches of data for training/testing
from torchvision import datasets # popular datasets, architectures and common image transformations
from torchvision.transforms import ToTensor # an easy way to convert PIL images to Tensors

import pandas as pd
import numpy as np # NumPy, the python array library
import random # for generating random numbers
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
import time

# For utils
from tqdm.notebook import tqdm
import time as time
import sys # to stop the execution of the code
from IPython.display import display, Javascript # to save 
import pygame # to sound
import csv


# software modules
import models_regression as Models
import utils
import dataset as Dataset


# -------- Structures --------
class Individual:
    def __init__(self, structure_gene, af_gene, lr_gene, dropout_gene ,fitness=0.0):
        """
        Initialize an individual with a gene, solution, and fitness.
        
        Parameters:
        gene (float): The value in the space of the gene.
        solution (Any): The value in the space of the solution.
        fitness (float): The fitness value of the individual.
        """
        self.structure_gene = structure_gene
        self.af_gene = af_gene
        self.lr_gene = lr_gene
        self.dropout_gene = dropout_gene

        self.structure_solution = self.from_structureGene_to_structureSolution(structure_gene)  
        self.af_solution = self.af_gene
        self.lr_solution = self.lr_gene
        self.dropout_solution = self.dropout_gene

        self.fitness = 0.0
    
    def from_structureGene_to_structureSolution(self, structureGenotype):
        """
        Convert the gene to a solution.
        """
        # For all element in the list of the genes, the solution is the list of the integer raund of the gene
        return [round(g) for g in structureGenotype]
    
    def __repr__(self):
        return f"Individual: nn_gene = {self.structure_gene}, af = {self.af_gene}, lr = {self.lr_gene}, dropout = {self.dropout_gene} | nn_sol = {self.structure_solution}, Fitness = {self.fitness}"
    

# -------- Initialization --------

def initialization(
        initial_population_size,
        number_of_layers_interval,
        number_of_units_interval,
        activation_functions,
        learning_rate_interval,
        dropout_interval):
    
    structure_genes = []
    
    for _ in range(initial_population_size):
        num_layers = random.randint(number_of_layers_interval[0], number_of_layers_interval[1])
        structure_gene = [random.randint(number_of_units_interval[0], number_of_units_interval[1]) for _ in range(num_layers)]
        structure_genes.append(structure_gene)

    af_genes = [random.choice(activation_functions) for _ in range(initial_population_size)]

    lr_genes = [random.uniform(learning_rate_interval[0], learning_rate_interval[1]) for _ in range(initial_population_size)]

    dropout_genes = [random.uniform(dropout_interval[0], dropout_interval[1]) for _ in range(initial_population_size)]
    
    individuals = [Individual(structure_gene, af_gene, lr_gene, dropout_gene) for structure_gene, af_gene, lr_gene, dropout_gene in zip(structure_genes, af_genes, lr_genes, dropout_genes)]

    return individuals




# -------- Evaluation --------
def fitness_evaluation(
        individual,
        training_data,
        validation_data,
        num_epochs_per_evaluation,
        #lr,
        input_size, # 40
        #hidden_sizes,
        output_size, # 15
        #activation_function,
        #initialize_weights,
        #dropout,
        device): 

    params = {
        'input_size': input_size,
        'hidden_layer_size': individual.structure_solution[0],
        'num_layers': 2,
        'output_size': output_size,
        'dropout': individual.dropout_solution,
    }
    model = Models.LSTM1(**params).to(device)

    # Optimizer parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=individual.lr_solution)
    criterion = nn.MSELoss()

    training_info = utils.ModelTools.train(
        model_id = 'LSTM1',
        model = model,
        criterion = criterion,
        optimizer = optimizer,
        train_loader = training_data,
        val_loader = validation_data,
        n_epochs = num_epochs_per_evaluation,
        save = False,
        device = device)

    return np.mean(training_info['val_loss'][-5:])



def evaluate_individuals(
        individuals,
        training_data,
        validation_data,
        num_epochs_per_evaluation,
        #lr,
        input_size, # 40
        #hidden_sizes,
        output_size, # 15
        #activation_function,
        #initialize_weights,
        #dropout,
        device): 
    
    for individual in individuals:
        individual.fitness = fitness_evaluation(
            individual = individual,
            training_data = training_data,
            validation_data = validation_data,
            num_epochs_per_evaluation = num_epochs_per_evaluation,
            #lr,
            input_size = input_size, # 40
            #hidden_sizes,
            output_size = output_size, # 15
            #activation_function,
            #initialize_weights,
            #dropout,
            device = device)
        

# -------- Selection --------

def find_best_individual(population, maximize):
    """
    Find the individual with the best fitness value in the population.
    
    Parameters:
    population (list): A list of Individual objects.
    
    Returns:
    Individual: The individual with the best fitness value.
    """
    if maximize:
        best_individual = max(population, key=lambda x: x.fitness)
    else:
        best_individual = min(population, key=lambda x: x.fitness)
    return best_individual




# Tournament selection
# number_of_selected: The number of individuals to be selected.
# percentage_of_population_size: The percentage of the population size to use for the tournament.
# heterogeneity = True: If True, the selected individuals must be unique.
def tournament_selection(population, number_of_selected, percentage_of_population_size, heterogeneity):
    selected_individuals = []
    tournament_size = max(1, int(len(population) * percentage_of_population_size))

    for _ in range(number_of_selected):
        if heterogeneity:
            # Ensure uniqueness
            best_individual = None
            while best_individual is None or best_individual in selected_individuals:
                # Randomly select individuals for the tournament
                tournament_individuals = random.sample(population, tournament_size)
                # Select the best individual from the tournament
                best_individual = min(tournament_individuals, key=lambda ind: ind.fitness)
        else:
            # Randomly select individuals for the tournament
            tournament_individuals = random.sample(population, tournament_size)
            # Select the best individual from the tournament
            best_individual = min(tournament_individuals, key=lambda ind: ind.fitness)

        selected_individuals.append(best_individual)

    return selected_individuals




# Truncated selection
# k: The number of individuals to be selected.
# maximize: If True, select individuals with greatest fitness; otherwise, select individuals with smallest fitness.
def truncated_selection(population, k, maximize):
    # Sort the population based on fitness
    sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=maximize)
    
    # Select the top k individuals
    selected_individuals = sorted_population[:k]
    
    return selected_individuals



# -------- Evolutionary operators --------
def correct_layer_size(gene, lb, ub):
    if gene < lb or gene > ub:
        if gene < lb:
            gene = lb
        else:
            gene = ub
    return gene


def correct_layers_size(genotype, lb, ub):
    corrected_genotype = [lb if gene < lb else ub if gene > ub else gene for gene in genotype]
    return corrected_genotype


def correct_general_size(value, ub, lb):
    if value < lb:
        value = lb
    elif value > ub:
        value = ub
    return value


# Description:
# aaa|aa 
# bbb|bb
# --> aaa|bb, bbb|aa
def structure_singlePointCX(parent1, parent2):
    gene1, gene2 = parent1.structure_gene, parent2.structure_gene
    af_gene1, af_gene2 = parent1.af_gene, parent2.af_gene
    lr_gene1, lr_gene2 = parent1.lr_gene, parent2.lr_gene
    if len(gene1) == 1 or len(gene2) == 1:
        return parent1, parent2  # No crossover if a parent has only one layer
    point = random.randint(1, min(len(gene1), len(gene2)) - 1)
    child_gene1 = gene1[:point] + gene2[point:]
    child_gene2 = gene2[:point] + gene1[point:]
    child1 = Individual(structure_gene=child_gene1, af_gene=af_gene1, lr_gene=lr_gene1, dropout_gene=parent1.dropout_gene)
    child2 = Individual(structure_gene=child_gene2, af_gene=af_gene2, lr_gene=lr_gene2, dropout_gene=parent2.dropout_gene)
    return child1, child2



def structure_SPC_E(parent1, parent2):
    gene1, gene2 = parent1.structure_gene, parent2.structure_gene
    af_gene1, af_gene2 = parent1.af_gene, parent2.af_gene
    lr_gene1, lr_gene2 = parent1.lr_gene, parent2.lr_gene
    
    # Determine the fittest parent
    if parent1.fitness < parent2.fitness:
        fitter_parent = parent1
        other_parent = parent2
    else:
        fitter_parent = parent2
        other_parent = parent1

    if len(gene1) == 1 or len(gene2) == 1:
        return parent1, parent2  # No crossover if a parent has only one layer
    
    point = random.randint(1, min(len(gene1), len(gene2)) - 1)
    
    # Split genes at the crossover point
    first_part_fitter = fitter_parent.structure_gene[:point]
    second_part_fitter = fitter_parent.structure_gene[point:]
    
    first_part_other = other_parent.structure_gene[:point]
    second_part_other = other_parent.structure_gene[point:]
    
    # Compute averages for the second part of genes
    avg_second_part = [(a + b) / 2 for a, b in zip(second_part_fitter, second_part_other)]
    avg_first_part = [(a + b) / 2 for a, b in zip(first_part_fitter, first_part_other)]
    
    # Create new genes for the children
    child_gene1 = first_part_fitter + avg_second_part
    child_gene2 = avg_first_part + second_part_fitter
    
    # Create child individuals
    child1 = Individual(structure_gene=child_gene1, af_gene=af_gene1, lr_gene =lr_gene1, dropout_gene=parent1.dropout_gene)
    child2 = Individual(structure_gene=child_gene2, af_gene=af_gene2, lr_gene =lr_gene2, dropout_gene=parent2.dropout_gene)
    
    return child1, child2



# k = 0.5 is good, k = 0 is intermediate recombination 
def structure_SPC_lineRecombination(parent1, parent2, k, lb, ub):
    gene1, gene2 = parent1.structure_gene, parent2.structure_gene
    af_gene1, af_gene2 = parent1.af_gene, parent2.af_gene
    lr_gene1, lr_gene2 = parent1.lr_gene, parent2.lr_gene
    
    # Determine the fittest parent
    if parent1.fitness < parent2.fitness:
        fitter_parent = parent1
        other_parent = parent2
    else:
        fitter_parent = parent2
        other_parent = parent1

    if len(gene1) == 1 or len(gene2) == 1:
        return parent1, parent2  # No crossover if a parent has only one layer
    
    point = random.randint(1, min(len(gene1), len(gene2)) - 1)
    
    # Split genes at the crossover point
    first_part_fitter = fitter_parent.structure_gene[:point]
    second_part_fitter = fitter_parent.structure_gene[point:]
    
    first_part_other = other_parent.structure_gene[:point]
    second_part_other = other_parent.structure_gene[point:]
    
    # Compute averages for the second part of genes

    max_len_first_part = max(len(first_part_fitter), len(first_part_other))
    max_len_second_part = max(len(second_part_fitter), len(second_part_other))
    ks_first_part =  [random.uniform(-k, 1 + k) for _ in range(max_len_first_part)]
    ks_second_part =  [random.uniform(-k, 1 + k) for _ in range(max_len_second_part)]
    
    line_recombination_first_part = [
        kk * a + (1 - kk) * b
        for a, b, kk in zip(first_part_other, first_part_fitter, ks_first_part)]
    line_recombination_second_part = [
        kk * a + (1 - kk) * b
        for a, b, kk in zip(second_part_other, second_part_fitter, ks_second_part)]
    
    
    # Create new genes for the children
    child_gene1 = first_part_fitter + line_recombination_second_part
    child_gene2 = line_recombination_first_part + second_part_fitter

    child_gene1 = correct_layers_size(genotype=child_gene1, lb=lb, ub=ub)
    child_gene2 = correct_layers_size(genotype=child_gene2, lb=lb, ub=ub)
    
    # Create child individuals
    child1 = Individual(structure_gene=child_gene1, af_gene=af_gene1, lr_gene =lr_gene1, dropout_gene=parent1.dropout_gene)
    child2 = Individual(structure_gene=child_gene2, af_gene=af_gene2, lr_gene =lr_gene2, dropout_gene=parent2.dropout_gene)
    
    return child1, child2



# k = 0.5 is good, k = 0 is intermediate recombination 
def structure_lineRecombination_oneLayer(parent1, parent2, k, lb, ub):
    gene1, gene2 = parent1.structure_gene[0], parent2.structure_gene[0]
        
    # Exctract two random point in (-k, 1+k)
    k1 = random.uniform(-k, 1 + k)
    k2 = random.uniform(-k, 1 + k)
    
    mutated_gene1 = k1 * gene1 + (1 - k1) * gene2
    mutated_gene2 = k2 * gene1 + (1 - k2) * gene2

    # Correct layer size 
    mutated_gene1 = correct_layer_size(gene = mutated_gene1, lb = lb, ub = ub)
    mutated_gene2 = correct_layer_size(gene = mutated_gene2, lb = lb, ub = ub)
    
    # Create child individuals
    child1 = Individual(structure_gene=[mutated_gene1], af_gene=parent1.af_gene, lr_gene =parent1.lr_gene, dropout_gene=parent1.dropout_gene)
    child2 = Individual(structure_gene=[mutated_gene2], af_gene=parent2.af_gene, lr_gene =parent2.lr_gene, dropout_gene=parent2.dropout_gene)
    
    return child1, child2


def af_crossover(parent1, parent2, p_pick_the_best):

    # Find the af of the fittest parent
    if parent1.fitness < parent2.fitness:
        af1 = parent1.af_gene # af of the fittest parent
        af2 = parent2.af_gene
    else:
        af1 = parent2.af_gene
        af2 = parent1.af_gene

    # Define two individuals with a probability p to pick the af of the fittest parent
    if random.random() < p_pick_the_best:
        child1 = Individual(structure_gene=parent1.structure_gene, af_gene=af1, lr_gene=parent1.lr_gene, dropout_gene=parent1.dropout_gene)
        child2 = Individual(structure_gene=parent2.structure_gene, af_gene=af1, lr_gene=parent2.lr_gene, dropout_gene=parent2.dropout_gene)
    else:
        child1 = Individual(structure_gene=parent1.structure_gene, af_gene=af2, lr_gene=parent1.lr_gene, dropout_gene=parent1.dropout_gene)
        child2 = Individual(structure_gene=parent2.structure_gene, af_gene=af2, lr_gene=parent2.lr_gene, dropout_gene=parent2.dropout_gene)
    
    return child1, child2



def lr_linearRecombination(parent1, parent2, k, lb, ub):

    gene1 = parent1.lr_gene
    gene2 = parent2.lr_gene
    a1 = np.random.uniform(-k, 1 + k)
    a2 = np.random.uniform(-k, 1 + k)

    child_gene1 = a1 * gene1 + (1 - a1) * gene2
    child_gene2 = a2 * gene1 + (1 - a2) * gene2    

    # correct lr size 
    child_gene1 = correct_general_size(value = child_gene1, ub = ub, lb = lb)
    child_gene2 = correct_general_size(value = child_gene2, ub = ub, lb = lb)

    child1 = Individual(structure_gene=parent1.structure_gene, af_gene=parent1.af_gene, lr_gene=child_gene1, dropout_gene=parent1.dropout_gene)
    child2 = Individual(structure_gene=parent2.structure_gene, af_gene=parent2.af_gene, lr_gene=child_gene2, dropout_gene=parent2.dropout_gene)
                           
    return child1, child2



def dropout_linearRecombination(parent1, parent2, k, lb, ub):

    gene1 = parent1.dropout_gene
    gene2 = parent2.dropout_gene
    a1 = np.random.uniform(-k, 1 + k)
    a2 = np.random.uniform(-k, 1 + k)

    child_gene1 = a1 * gene1 + (1 - a1) * gene2
    child_gene2 = a2 * gene1 + (1 - a2) * gene2    

    # correct lr size 
    child_gene1 = correct_general_size(value = child_gene1, ub = ub, lb = lb)
    child_gene2 = correct_general_size(value = child_gene2, ub = ub, lb = lb)

    child1 = Individual(structure_gene=parent1.structure_gene, af_gene=parent1.af_gene, lr_gene=parent1.lr_gene, dropout_gene=child_gene1)
    child2 = Individual(structure_gene=parent2.structure_gene, af_gene=parent2.af_gene, lr_gene=parent2.lr_gene, dropout_gene=child_gene2)
                           
    return child1, child2



def structure_PP_selfAdaptiveMutation(population):
    # Find the maximum lenght of the genes of the population 
    max_len = max([len(ind.structure_gene) for ind in population])

    # Find the mean of each gene of the genotype
    genotype_mean = [np.mean([ind.structure_gene[i] for ind in population if i < len(ind.structure_gene)]) for i in range(max_len)]

    # Find the maximum of each gene of the genotype
    genotype_max = [np.max([ind.structure_gene[i] for ind in population if i < len(ind.structure_gene)]) for i in range(max_len)]

    return genotype_mean, genotype_max


def structure_selfAdaptiveGaussianMutation(parent, genotype_mean, lb, ub):
    gene = parent.structure_gene
    mutated_gene = []
    for i in range(len(gene)):
        mutated_gene.append(gene[i] + np.random.normal(0, genotype_mean[i]/10))
        mutated_gene[i] = correct_layer_size(gene=mutated_gene[i], lb=lb, ub=ub)
    return Individual(structure_gene=mutated_gene, af_gene=parent.af_gene, lr_gene=parent.lr_gene, dropout_gene=parent.dropout_gene)  


def af_mutation(parent, activation_functions, p_change_af):
    gene = parent.af_gene
    if random.random() < p_change_af:
        gene = random.choice(activation_functions)
    return Individual(structure_gene=parent.structure_gene, af_gene=gene, lr_gene=parent.lr_gene, dropout_gene=parent.dropout_gene)



def lr_PP_selfAdaptiveMutation(population):
    # Find the mean of the learning rates of the population
    lr_mean = np.mean([ind.lr_gene for ind in population])
    return lr_mean



def dropout_PP_selfAdaptiveMutation(population):
    # Find the mean of the learning rates of the population
    dropout_mean = np.mean([ind.dropout_gene for ind in population])
    return dropout_mean



def lr_selfAdaptiveGaussianMutation(parent, lr_mean, lb, ub):
    gene = parent.lr_gene
    mutated_gene = gene + np.random.normal(0, lr_mean/10)
    mutated_gene = correct_general_size(value=mutated_gene, lb=lb, ub=ub)
    return Individual(structure_gene=parent.structure_gene, af_gene=parent.af_gene, lr_gene=mutated_gene, dropout_gene=parent.dropout_gene)



def dropout_selfAdaptiveGaussianMutation(parent, dropout_mean, lb, ub):
    gene = parent.dropout_gene
    mutated_gene = gene + np.random.normal(0, dropout_mean/10)
    mutated_gene = correct_general_size(value=mutated_gene, lb=lb, ub=ub)
    return Individual(structure_gene=parent.structure_gene, af_gene=parent.af_gene, lr_gene=parent.lr_gene, dropout_gene=mutated_gene)



# This mutation removes a gene
def structure_shrink_mutation(individual):
    mutated_gene = individual.structure_gene.copy()
    if len(mutated_gene) <= 1:
        return Individual(structure_gene=mutated_gene, af_gene=individual.af_gene, lr_gene=individual.lr_gene, dropout_gene=individual.dropout_gene)
    
    index = random.randint(0, len(mutated_gene) - 1)
    mutated_gene.pop(index)
    
    return Individual(structure_gene=mutated_gene, af_gene=individual.af_gene, lr_gene=individual.lr_gene, dropout_gene=individual.dropout_gene)  



# This mutation adds a gene
def structure_gaussianGrowMutation(individual, genotype_mean, position, lb, ub):
    mutated_gene = individual.structure_gene.copy()
    
    # Determine the index to insert the new gene 
    if position == 'beginning':
        index = 0
    elif position == 'end':
        index = len(mutated_gene)
    else:  # Default to inserting at a random position
        index = random.randint(0, len(mutated_gene))

    # If condition 
    # It can happen that the new layer is the last one AND the largest individual has less number of layer 
    # So, we don't have data on the average of the population for that position  
    if index == len(mutated_gene) and len(mutated_gene) >= len(genotype_mean):
        inspiration_index = random.randint(0, len(genotype_mean) - 1) # Randomly select an index from the genotype
        new_layer_size = np.random.normal(genotype_mean[inspiration_index], genotype_mean[inspiration_index]/10)
        new_layer_size = correct_layer_size(gene=new_layer_size, lb=lb, ub=ub)
        mutated_gene.insert(index, new_layer_size)
    else:
        new_layer_size = np.random.normal(genotype_mean[index], genotype_mean[index]/10)
        new_layer_size = correct_layer_size(gene=new_layer_size, lb=lb, ub=ub)
        mutated_gene.insert(index, new_layer_size)
    
    return Individual(structure_gene=mutated_gene, af_gene=individual.af_gene, lr_gene=individual.lr_gene, dropout_gene=individual.dropout_gene)




# -------- Generate offsprings --------

def generate_offsprings(
        population,
        number_of_offsprings,
        tournament_percentage,
        p_crossover_vs_mutation,
        structure_k_crossover,
        structure_lb_size,
        structure_ub_size,
        #af_p_pick_the_best,
        lr_k,
        dropout_k,
        #p_changeLength_mutation,
        #structure_grow_position,
        #activation_functions,
        #af_p_change_af,
        lr_lb,
        lr_ub,
        dropout_lb,
        dropout_ub):
    
    offsprings = []

    # Pre-process the population
    structure_genotype_mean, _ = structure_PP_selfAdaptiveMutation(population=population)
    lr_genotype_mean = lr_PP_selfAdaptiveMutation(population=population)
    dropout_genotype_mean = dropout_PP_selfAdaptiveMutation(population=population)

    for _ in range(number_of_offsprings // 2):
        # Select parents for crossover
        parent1, parent2 = tournament_selection(population = population, number_of_selected=2, percentage_of_population_size=tournament_percentage, heterogeneity=True)

        if random.random() < p_crossover_vs_mutation:
            offspring1, offspring2 = structure_lineRecombination_oneLayer(parent1=parent1, parent2=parent2, k=structure_k_crossover, lb=structure_lb_size, ub=structure_ub_size)
            offspring1, offspring2 = lr_linearRecombination(parent1=offspring1, parent2=offspring2, k = lr_k, lb = lr_lb, ub =lr_ub)
            offspring1, offspring2 = dropout_linearRecombination(parent1=offspring1, parent2=offspring2, k = dropout_k, lb = dropout_lb, ub =dropout_ub)
        else:
            offspring1 = structure_selfAdaptiveGaussianMutation(parent = parent1, genotype_mean=structure_genotype_mean, lb = structure_lb_size, ub = structure_ub_size)
            offspring2 = structure_selfAdaptiveGaussianMutation(parent = parent2, genotype_mean=structure_genotype_mean, lb = structure_lb_size, ub = structure_ub_size)

            offspring1 = lr_selfAdaptiveGaussianMutation(parent = offspring1, lr_mean = lr_genotype_mean, lb = lr_lb, ub =lr_ub)
            offspring2 = lr_selfAdaptiveGaussianMutation(parent = offspring2, lr_mean = lr_genotype_mean, lb = lr_lb, ub =lr_ub)

            offspring1 = dropout_selfAdaptiveGaussianMutation(parent = offspring1, dropout_mean = dropout_genotype_mean, lb = dropout_lb, ub =dropout_ub)
            offspring2 = dropout_selfAdaptiveGaussianMutation(parent = offspring2, dropout_mean = dropout_genotype_mean, lb = dropout_lb, ub =dropout_ub)

        # Add the offspring to the population
        offsprings.append(offspring1)
        offsprings.append(offspring2)
            
    # If it is odd
    if number_of_offsprings % 2 == 1:
        parent = tournament_selection(population = population, number_of_selected = 1, percentage_of_population_size = tournament_percentage, heterogeneity=False)[0]
        offspring = structure_selfAdaptiveGaussianMutation(parent = parent, genotype_mean = structure_genotype_mean, lb = structure_lb_size, ub = structure_ub_size)
        offspring = lr_selfAdaptiveGaussianMutation(parent = offspring, lr_mean = lr_genotype_mean, lb = lr_lb, ub =lr_ub)
        offspring = dropout_selfAdaptiveGaussianMutation(parent = offspring, dropout_mean = dropout_genotype_mean, lb = dropout_lb, ub =dropout_ub)
        offsprings.append(offspring)    

    return offsprings




# -------- Complete algorithm --------
# Es lambda mu for structure (s), learning rate (l) and dropout (d) in the classification case

def es_lambdaMu_sld_regression(
        max_fitness_evaluations,
        mu_population_size,
        init_layersNumber_interval,
        init_unitsNumber_interval,
        init_activation_functions,
        init_lr_interval,
        init_dropout_interval,
        nn_training_data,
        nn_validation_data,
        num_epochs_per_nn_evaluation,
        nn_input_size,
        nn_output_size,
        device,
        maximize_fitness,
        lambda_offsprings_size,
        tornamentPercentage_for_offspringGeneration,
        p_crossover_vs_mutation,
        structure_k_value_for_crossover,
        structure_lb_size,
        structure_ub_size,
        lr_k,
        dropout_k,
        lr_lb,
        lr_ub,
        dropout_lb,
        dropout_ub,
        csv_file_path):

    # Initialize the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Generation", "Fitness Evaluations", "Time elapsed", "Best Individual", 
            "Population (sorted by fitness)"
        ])
    
        # Initialize the population
        fitness_evaluations = 0
        gen_index = 1
        time_start = time.time()

        population = initialization(
            initial_population_size=mu_population_size,
            number_of_layers_interval=init_layersNumber_interval,
            number_of_units_interval=init_unitsNumber_interval,
            activation_functions=init_activation_functions,
            learning_rate_interval=init_lr_interval,
            dropout_interval=init_dropout_interval
        )

        # Evaluate the individuals
        evaluate_individuals(
            individuals=population,
            training_data=nn_training_data,
            validation_data=nn_validation_data,
            num_epochs_per_evaluation=num_epochs_per_nn_evaluation,
            input_size=nn_input_size,
            output_size=nn_output_size,
            device=device
        )

        fitness_evaluations += mu_population_size

        # Compute the estimated time
        time_first_pop = time.time()

        estimated_seconds = ((time_first_pop - time_start) / mu_population_size) * max_fitness_evaluations
        estimated_hours = estimated_seconds / 3600
        estimated_remaining_minutes = (estimated_hours - int(estimated_hours)) * 60
        print(f"Estimated time: {int(estimated_hours)} hours and {int(estimated_remaining_minutes)} minutes")

        # Print first results
        best_individuals = []
        best_individual = find_best_individual(population=population, maximize=maximize_fitness)
        best_individuals.append(best_individual)

        # Write to CSV
        writer.writerow([
            gen_index, fitness_evaluations,  time.time() - time_start, best_individual,
            [str(ind) for ind in sorted(population, key=lambda ind: ind.fitness)]
        ])

        # Evolution loop
        while fitness_evaluations < max_fitness_evaluations:

            # Select offsprings
            offsprings = generate_offsprings(
                population=population,
                number_of_offsprings=lambda_offsprings_size,
                tournament_percentage=tornamentPercentage_for_offspringGeneration,
                p_crossover_vs_mutation=p_crossover_vs_mutation,
                structure_k_crossover=structure_k_value_for_crossover,
                structure_lb_size=structure_lb_size,
                structure_ub_size=structure_ub_size,
                lr_k=lr_k,
                dropout_k=dropout_k,
                lr_lb=lr_lb,
                lr_ub=lr_ub,
                dropout_lb=dropout_lb,
                dropout_ub=dropout_ub
            )

            # Evaluate the individuals
            evaluate_individuals(
                individuals=offsprings,
                training_data=nn_training_data,
                validation_data=nn_validation_data,
                num_epochs_per_evaluation=num_epochs_per_nn_evaluation,
                input_size=nn_input_size,
                output_size=nn_output_size,
                device=device
            )

            fitness_evaluations += lambda_offsprings_size

            # Build new population
            population = truncated_selection(
                population=offsprings,
                k=mu_population_size,
                maximize=maximize_fitness
            )

            # Update the best individuals and write to CSV
            best_individual = find_best_individual(population=population, maximize=maximize_fitness)
            best_individuals.append(best_individual)
            writer.writerow([
                gen_index + 1, fitness_evaluations, time.time() - time_start, best_individual, 
                [str(ind) for ind in population]
            ])
            gen_index += 1

        # Compute the best individual ever
        best_individual_ever = find_best_individual(population=best_individuals, maximize=maximize_fitness)
        writer.writerow([
                "Best individual ever: ", None , None, best_individual_ever
            ])

    return best_individual_ever




