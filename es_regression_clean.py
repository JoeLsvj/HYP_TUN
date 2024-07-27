#!/usr/bin/env python
# coding: utf-8

# # ES for ht for classification 
# 
# HT to tune 
# - lr 
# - dropout 
# - number of units 
# - activation function 
# 
# 
# 

# ## The NN

# Libs

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
import csv

# software modules
import models_regression as Models
import utils
import dataset as Dataset

# Set seeds

SEED = 0 # seed for reproducibility

random.seed(SEED) # set the seed for random numbers
np.random.seed(SEED) # set the seed for numpy arrays


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use the GPU if available
print(f"device: {device}")

dataset = pd.read_parquet('/home/joel/Desktop/DSSC/DEEP_LEARNING/stocks_forecasting_LOB/Data/FI-2010-Cleaned/NoAuction/Zscore')
dataset = dataset.astype(float)
#display(dataset)

target_col = 'P_Ask_1'
target_cols = []
shifts = [ 1, 5, 10, 20, 50 ]

for shift in shifts:
    colname = f'Target_{shift}'
    target_cols.append(colname)
    dataset[colname] = dataset[target_col].shift(-shift)
dataset.dropna(inplace=True)


# Split the dataframe into train, validation and test
dataset_train, dataset_val, dataset_test = utils.DataTools.train_val_test_split(dataset)

# Standardize the data
#scaler = preprocessing.StandardScaler()
#scaler.fit(dataset_train)
#dataset_train[dataset_train.columns] = scaler.transform(dataset_train)
#dataset_val[dataset_val.columns] = scaler.transform(dataset_val)
#dataset_test[dataset_test.columns] = scaler.transform(dataset_test)

means = dataset_train.mean(axis=0)
stds = dataset_train.std(axis=0)

means[-len(target_cols):] = means[target_cols]
stds[-len(target_cols):] = stds[target_cols]

dataset_train = ( dataset_train - means ) / stds
dataset_val = ( dataset_val - means ) / stds
dataset_test = ( dataset_test - means ) / stds


# Split covariates and response variables
X_train, y_train = utils.DataTools.split_x_y(dataset_train, target_cols)
X_val, y_val = utils.DataTools.split_x_y(dataset_val, target_cols)
X_test, y_test = utils.DataTools.split_x_y(dataset_test, target_cols)

# Convert to pytorch tensors
X_train, X_val, X_test = utils.DataTools.numpy_to_tensor( X_train, X_val, X_test, dtype=torch.float32 )
y_train, y_val, y_test = utils.DataTools.numpy_to_tensor( y_train, y_val, y_test, dtype=torch.float32 )

# Create Torch Dataset
lookback_period = 100
dataset_train = Dataset.TimeSeriesDataset(
    X         = X_train,
    y         = y_train,
    seq_len_x = lookback_period,
    seq_len_y = 1,
    offset    = lookback_period-1,
    channels  = False,
    task      = Dataset.TimeSeriesDataset.Task.REGRESSION
)
dataset_val = Dataset.TimeSeriesDataset(
    X         = X_val,
    y         = y_val,
    seq_len_x = lookback_period,
    seq_len_y = 1,
    offset    = lookback_period-1,
    channels  = False,
    task      = Dataset.TimeSeriesDataset.Task.REGRESSION
)
dataset_test = Dataset.TimeSeriesDataset(
    X         = X_test,
    y         = y_test,
    seq_len_x = lookback_period,
    seq_len_y = 1,
    offset    = lookback_period-1,
    channels  = False,
    task      = Dataset.TimeSeriesDataset.Task.REGRESSION
)

# Create Torch DataLoader
batch_size = 32
dataloader_train = Dataset.TimeSeriesLoader(
    dataset = dataset_train,
    batch_size = batch_size,
    shuffle = True
)
dataloader_val = Dataset.TimeSeriesLoader(
    dataset = dataset_val,
    batch_size = batch_size,
    shuffle = False
)
dataloader_test = Dataset.TimeSeriesLoader(
    dataset = dataset_test,
    batch_size = batch_size,
    shuffle = False
)

#print(f'Train dataset: {len(dataset_train)} samples')
#print(f'Test dataset: {len(dataset_test)} samples')

dataloader_train.dataset.channels = False
dataloader_val.dataset.channels = False
dataloader_test.dataset.channels = False


# The EA: Complete algorithm

import es_regression_functions as esr

# Example usage
max_fitness_evaluations = 55 # Just for trial
num_epochs_per_nn_evaluation = 20

mu_population_size = 10
lambda_offsprings_size = 15

tornamentPercentage_for_offspringGeneration = 0.2
p_crossover_vs_mutation = 0.75

init_layersNumber_interval = [1,1]
init_unitsNumber_interval = [10,100]
init_activation_functions = [nn.ReLU] 
init_lr_interval = (0.0001, 0.1)
init_dropout_interval = (0.0, 0.2)
nn_training_data = dataloader_train
nn_validation_data = dataloader_val
nn_input_size = X_train.shape[1]
nn_output_size = y_test.shape[1]
maximize_fitness = False
structure_k_value_for_crossover = 0
structure_lb_size = 10
structure_ub_size = np.inf
lr_k = 0
dropout_k = 0
lr_lb = 0.0001
lr_ub = 0.1
dropout_lb = 0.0
dropout_ub = 0.2
csv_file_path = '/home/joel/Desktop/DSSC/DEEP_LEARNING/stocks_forecasting/Samuele_work/EA-for-hyperparamters-tuning/es_regression/csv_results/LSTM1_HYP.csv'

best_individual_ever = esr.es_lambdaMu_sld_regression(
    max_fitness_evaluations = max_fitness_evaluations,
    mu_population_size = mu_population_size,
    init_layersNumber_interval = init_layersNumber_interval,
    init_unitsNumber_interval = init_unitsNumber_interval,
    init_activation_functions = init_activation_functions,
    init_lr_interval = init_lr_interval,
    init_dropout_interval = init_dropout_interval,
    nn_training_data = nn_training_data,
    nn_validation_data = nn_validation_data,
    num_epochs_per_nn_evaluation = num_epochs_per_nn_evaluation,
    nn_input_size = nn_input_size,
    nn_output_size = nn_output_size,
    device = device,
    maximize_fitness = maximize_fitness,
    lambda_offsprings_size = lambda_offsprings_size,
    tornamentPercentage_for_offspringGeneration = tornamentPercentage_for_offspringGeneration,
    p_crossover_vs_mutation = p_crossover_vs_mutation,
    structure_k_value_for_crossover = structure_k_value_for_crossover,
    structure_lb_size = structure_lb_size,
    structure_ub_size = structure_ub_size,
    lr_k = lr_k,
    dropout_k = dropout_k,
    lr_lb = lr_lb,
    lr_ub = lr_ub,
    dropout_lb = dropout_lb,
    dropout_ub = dropout_ub,
    csv_file_path=csv_file_path)