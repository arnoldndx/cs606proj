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


# Setting up arguments
parser = argparse.ArgumentParser()
parser.add_argument("--file", type = str, default = 'test', help = "Filename prefix. "
                                                                          "You should give a meaningful name for easy tracking.")
parser.add_argument('--weights', type = str, default = 'defined', choices=['defined', 'trained'])
parser.add_argument('--weights_data', type = str, default = "../data/soft_constraint_weights_temp.csv",  help = 'Filepath for weights training data. Should be either a csv file for defined weights,'
                                                        'or a folder containing midi files for trained wweights. Data should be in ../data/')
parser.add_argument('--hard_constraints_choice', type = str, default = '../data/hard_constraint_choice.csv', help = 'Filepath for hard constraint choices')
parser.add_argument('--time_limit', type = int, default = 600, help = 'Time limit for iterations (MP/CP) or Iteration limit for ALNS')

# Starting up
args = parser.parse_args()
models = ['mp','cp','alns']

model_dict = {'mp':'Mixed-integer programming',
              'cp':'Constraint programming',
              'alns': 'Adaptive LNS',
              'ga': 'Genetic algorithm'}

inputs = '../data/sample_input.csv'

titles = ['Minor - 4 note',
          'Minor - 8 note',
          'Minor - 16 note',
          'Major - 4 note',
          'Major - 8 note',
          'Major - 16 note',
          ]

results_comparison = {}

for i in range(-6,0):
    for model in models:
        try:
            progress_array = run_harmony_gen(model,args.file,args.weights,args.weights_data,args.hard_constraints_choice,args.time_limit,inputs,i)
            results_comparison[(model_dict[model],titles[i+6])] = progress_array
        except:
            print('=================== no feasible solution')
            results_comparison[(model_dict[model],titles[i+6])] = 'the model was note able to produce a feasible solution'
            pass
        
# plot the results
fig, axs = plt.subplots(3,2,figsize = (15,10))

i = 0
j = 0
title = titles[0]

# plot the results
fig, axs = plt.subplots(3,2,figsize = (15,10))

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

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

    
fig_fname = os.path.join('../outputs',
                         f"{args.file}_{f_datetime}.png")

fig.savefig(fig_fname, dpi=200)