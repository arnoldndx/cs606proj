# -*- coding: utf-8 -*-
'''
Created on Mon Feb 28 20:17:15 2022

These functions are for determining the soft constraints (if required)

Dummy code to define classes and functions from the assignment 2 script. Start with this first.

'''
import numpy as np
import numpy.random as rnd
import networkx as nx
import matplotlib.pyplot as plt
from lxml import etree as LET

from evrp import *
from pathlib import Path

import sys
sys.path.append('./ALNS')


### draw and output solution ###
def save_output(YourName, evrp, suffix):
    '''Draw the EVRP instance and save the solution
    Args:
        YourName::str
            your name, eg. John_Doe
        evrp::EVRP
            an EVRP object
        suffix::str
            suffix of the output file, 
            'initial' for random initialization
            and 'solution' for the final solution
    '''
    draw_evrp(YourName, evrp, suffix)
    generate_output(YourName, evrp, suffix)

### visualize EVRP ###
def create_graph(evrp):
    '''Create a directional graph from the EVRP instance
    Args:
        evrp::EVRP
            an EVRP object
    Returns:
        g::nx.DiGraph
            a directed graph
    '''
    g = nx.DiGraph(directed=True)
    g.add_node(evrp.depot.id, pos=(evrp.depot.x, evrp.depot.y), type=evrp.depot.type)
    for c in evrp.customers:
        g.add_node(c.id, pos=(c.x, c.y), type=c.type)
    for cs in evrp.CSs:
        g.add_node(cs.id, pos=(cs.x, cs.y), type=cs.type)
    return g

def draw_evrp(YourName, evrp, suffix):
    '''Draw the EVRP instance and the solution
    Args:
        YourName::str
            your name, eg. John_Doe
        evrp::EVRP
            an EVRP object
        suffix::str
            suffix of the output file, 
            eg. 'initial' for random initialization
            and 'solution' for the final solution
    '''
    g = create_graph(evrp)
    route = list(node.id for node in sum(evrp.route, []))
    edges = [(route[i], route[i+1]) for i in range(len(route) - 1) if route[i] != route[i+1]]
    g.add_edges_from(edges)
    colors = []
    for n in g.nodes:
        if g.nodes[n]['type'] == 0:
            colors.append('#0000FF')
        elif g.nodes[n]['type'] == 1:
            colors.append('#FF0000')
        else:
            colors.append('#00FF00')
    pos = nx.get_node_attributes(g, 'pos')
    fig, ax = plt.subplots(figsize=(24, 12))
    nx.draw(g, pos, node_color=colors, with_labels=True, ax=ax, 
            arrows=True, arrowstyle='-|>', arrowsize=12, 
            connectionstyle='arc3, rad = 0.025')

    plt.text(0, 6, YourName, fontsize=12)
    plt.text(0, 3, 'Instance: {}'.format(evrp.name), fontsize=12)
    plt.text(0, 0, 'Objective: {}'.format(evrp.objective()), fontsize=12)
    plt.savefig('{}_{}_{}.jpg'.format(YourName, evrp.name, suffix), dpi=300, bbox_inches='tight')
    
### generate output file for the solution ###
def generate_output(YourName, evrp, suffix):
    '''Generate output file (.txt) for the evrp solution, containing the instance name, the objective value, and the route
    Args:
        YourName::str
            your name, eg. John_Doe
        evrp::EVRP
            an EVRP object
        suffix::str
            suffix of the output file,
            eg. 'initial' for random initialization
            and 'solution' for the final solution
    '''
    str_builder = ['{}\nInstance: {}\nObjective: {}\n'.format(YourName, evrp.name, evrp.objective())]
    for idx, r in enumerate(evrp.route):
        str_builder.append('Route {}:'.format(idx))
        j = 0
        for node in r:
            if node.type == 0:
                str_builder.append('depot {}'.format(node.id))
            elif node.type == 1:
                str_builder.append('customer {}'.format(node.id))
            elif node.type == 2:
                str_builder.append('station {} Charge ({})'.format(node.id, evrp.vehicles[idx].battery_charged[j]))
                j += 1
        str_builder.append('\n')
    with open('{}_{}_{}.txt'.format(YourName, evrp.name, suffix), 'w') as f:
        f.write('\n'.join(str_builder))

### Destroy operators ###
# You can follow the example and implement destroy_2, destroy_3, etc
def destroy_1(current, random_state):
    ''' Destroy operator sample (name of the function is free to change)
    Args:
        current::EVRP
            an EVRP object before destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        destroyed::EVRP
            the evrp object after destroying
    '''
    destroyed = current.copy()
    # You should code here
    ...
    return destroyed

### Repair operators ###
# You can follow the example and implement repair_2, repair_3, etc
def repair_1(destroyed, random_state):
    ''' repair operator sample (name of the function is free to change)
    Args:
        destroyed::EVRP
            an EVRP object after destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        repaired::EVRP
            the evrp object after repairing
    '''
    # You should code here
    ...
    return repaired