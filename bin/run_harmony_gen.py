# -*- coding: utf-8 -*-
'''
Created on Mon Feb 28 20:17:15 2022

Dummy driver code from the assignment 2 script. Start with this first.

'''


import argparse
import numpy as np
import numpy.random as rnd
import networkx as nx
import matplotlib.pyplot as plt
from lxml import etree as LET

from evrp import *
from pathlib import Path

from src import midi_processing
from src import ml_modelling
from src import soft_constraints

import sys
sys.path.append('./ALNS')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='load data')
    parser.add_argument(dest='data', type=str, help='data')
    parser.add_argument(dest='seed', type=int, help='seed')
    args = parser.parse_args()
    
    # instance file and random seed
    xml_file = args.data
    seed = int(args.seed)
    
    # load data and random seed
    parsed = Parser(xml_file)
    evrp = EVRP(parsed.name, parsed.depot, parsed.customers, parsed.CSs, parsed.vehicle)
    
    # construct random initialized solution
    evrp.random_initialize(seed)
    print("Initial solution objective is {}.".format(evrp.objective()))
    
    # visualize initial solution and gernate output file
    save_output('YourName', evrp, 'initial')
    
    # ALNS
    random_state = rnd.RandomState(seed)
    alns = ALNS(random_state)
    # add destroy
    # You should add all your destroy and repair operators
    alns.add_destroy_operator(destroy_1)
    # add repair
    alns.add_repair_operator(repair_1)
    
    # run ALNS
    # select cirterion
    criterion = ...
    # assigning weights to methods
    omegas = [...]
    lambda_ = ...
    result = alns.iterate(evrp, omegas, lambda_, criterion,
                          iterations=1000, collect_stats=True)

    # result
    solution = result.best_state
    objective = solution.objective()
    print('Best heuristic objective is {}.'.format(objective))
    
    # visualize final solution and gernate output file
    save_output('YourName', solution, 'solution')