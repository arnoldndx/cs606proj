# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:40:46 2022

@author: Arnold
"""
#Standard Imports
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
# from collections import defaultdict
import timeit
import argparse
import logging
import copy
import numpy.random as rnd
import matplotlib.pyplot as plt

sys.path.append('../')

# Custom Imports
from src.run_harmony_gen_iterator import *

default_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Setting up arguments
parser = argparse.ArgumentParser()
parser.add_argument("--file", type = str, default = 'test', help = "Filename prefix. "
                                                                          "You should give a meaningful name for easy tracking.")
parser.add_argument('--weights', type = str, default = 'defined', choices=['defined', 'trained'])
parser.add_argument('--weights_data', type = str, default = "../data/soft_constraint_weights_temp.csv",  help = 'Filepath for weights training data. Should be either a csv file for defined weights,'
                                                        'or a folder containing midi files for trained wweights. Data should be in ../data/')
parser.add_argument('--hard_constraints_choice', type = str, default = '../data/hard_constraint_choice.csv', help = 'Filepath for hard constraint choices')
parser.add_argument('--time_limit', type = int, default = 600, help = 'Time limit for iterations (MP/CP) or Iteration limit for ALNS')
parser.add_argument('--input_melody', type = str, default = '../data/test_melody.mid', help = "Filepath for the input melody. Valid filetypes: .csv, .mid")
parser.add_argument('--max_generation', type = int, default = 300, help = 'number of generations to iterate through')
parser.add_argument('--population_size', type = int, default = 100, help = 'population size')
parser.add_argument('--mutation_probability', type = list, default = [0.6, 0.9], help = 'mutation probability = [lower_bound, higher_bound]')

# Starting up
args = parser.parse_args()
models = ['ga']

model_dict = {'ga': 'Genetic algorithm'}

inputs = '../data/sample_input.csv'

titles = ['Minor - 16 note', 'Major - 16 note']

generations = [50, 75, 100]

populations = [100, 250, 500]

mutation_probs = [[0.5, 0.9], [0.4, 0.9], [0.3, 0.9]]

results_comparison = {}

for i in range(-1, 0):
    for generation in generations:
        for population in populations:
           # try:
            progress_array = run_harmony_gen(
                'ga',
                args.file,
                args.weights,
                args.weights_data,
                args.hard_constraints_choice,
                args.time_limit,
                inputs,
                i,
                generation,
                population,
                args.mutation_probability,
            )
            results_comparison[((generation, population),titles[i])] = progress_array
            #except:
             #   print('=================== no feasible solution')
             #   results_comparison[((generation, population),titles[i])] = 'the model was note able to produce a feasible solution'
              #  pass
        
# plot the results
fig, axs = plt.subplots(3,3,figsize = (15,15))

i = 0
j = 0
title = titles[0]

# plot the results
fig, axs = plt.subplots(3,3,figsize = (15,15))

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

for i in range(3):
    axs[i,0].set_ylabel("Objective Value", fontsize=8)
    
for j in range(2):
    axs[2,j].set_xlabel("Time", fontsize=8)
    
i = 0
j = 0
title = titles[0]
axs[i,j].set_title(title, fontsize=8)

for result in results_comparison.keys():
    # set the index for the subplot to draw in
    if result[1] != title and i < 2:
        axs[i,j].legend()
        print(title,i,j)
        i += 1
        title = result[1]
        axs[i,j].set_title(title, fontsize=8)
    elif result[1] != title and i == 2:
        axs[i,j].legend()
        print(title,i,j)
        i = 0
        j += 1
        title = result[1]
        axs[i,j].set_title(title, fontsize=8)
    
    # create the x-s and y-s
    time = []
    obj_val = []
    if isinstance(results_comparison[result],str):
        pass
    else:
        for data in results_comparison[result]:
            time.append(data[0])
            obj_val.append(data[1])
        axs[i,j].plot(time, obj_val, label = result[0])

#set the last legend
axs[i,j].legend()

    
fig_fname = os.path.join(default_dir,
                         'outputs',
                         f"{args.file}_{datetime.now().strftime('%H%M_%d%m%Y')}.png")

fig.savefig(fig_fname, dpi=200)