'''
Description: ALNS + EVRP
Author: BO Jianyuan
Date: 2022-02-15 13:38:21
LastEditors: BO Jianyuan
LastEditTime: 2022-02-24 19:43:57
'''

#import argparse
import numpy as np
import numpy.random as rnd
import pandas as pd

from pathlib import Path

import sys
sys.path.append('./ALNS')



from alns import ALNS, State
from alns.criteria import HillClimbing, SimulatedAnnealing, RecordToRecordTravel

#Standard Imports

import sys
#import numpy as np

# from collections import defaultdict
#from docplex.cp.model import CpoModel

#Custom Imports
sys.path.append('../')
from src.chord import Chord

from src.musical_work_input import MusicalWorkInput,Harmony
#from src.cp_model import CPModel
from src.mp_model_for_ALNS_construction import MPModel
import src.evaluate

# Importing Musical Corpus
musical_work_df = pd.read_csv("../data/sample_input.csv")
musical_corpus = []
for i, title, meter, key, tonality, first_on_beat, melody in musical_work_df.itertuples():
    musical_corpus.append(MusicalWorkInput(title, meter, key, tonality, first_on_beat, [int(x) for x in melody.split(',')]))

# Defining dictionary of hard and soft constraint options:
hard_constraint_options = ['musical input', 'voice range', 'chord membership', 'first last chords',
                           'chord bass repetition', 'adjacent bar chords', 'voice crossing', 'parallel movement',
                          'chord spacing']

soft_constraint_options = ['chord progression', 'chord bass repetition', 'leap resolution',
                           'melodic movement', 'note repetition', 'parallel movement', 'voice overlap', 'adjacent bar chords',
                           'chord spacing', 'distinct notes', 'voice crossing', 'voice range']


# Model
#cp_model = CPModel("test", musical_corpus[0], chord_vocab)
music=musical_corpus[0]
print(music.title, music.key, music.tonality, 
      music.first_on_beat,music.melody, music.reference_note)
# Importing Chord Vocabulary
if music.tonality=="major":
    chord_df = pd.read_csv("../data/chord_vocabulary_major.csv", index_col = 0)
else:
    chord_df = pd.read_csv("../data/chord_vocabulary_minor.csv", index_col = 0)
chord_vocab = []
for name, note_intervals in chord_df.itertuples():
    chord_vocab.append(Chord(name, set(int(x) for x in note_intervals.split(','))))
    
# Defining dictionary of weights for each soft constraint options:
weight_df = pd.read_csv("../data/soft_constraint_weights.csv")
soft_constraint_w_weights={}
for _,name, w in weight_df.itertuples(): #name population is same as soft_constraint_options
    soft_constraint_w_weights[name]=float(w)
file_progression_cost="chord_progression_major_v1.csv" if music.tonality=="major" else "chord_progression_minor_v1.csv"
# Model
mp_model = MPModel("test", music,[], chord_vocab,
                    #hard_constraints, 
                    soft_constraint_w_weights, 
                    file_progression_cost=file_progression_cost)
solution = mp_model.solve()


df_soluiton=pd.DataFrame(np.array(solution))
df_soluiton.to_csv("ALNS_start.csv", index=False, header=False)


#%%
def destroy_1(current, random_state): ## worst sum-of-cost removal

    #class Harmony(State)
    
    destroyed = current.copy()   
    maxcost, destroy_idx=0
    for j in range(1, current.N-4):
        segment==current[:][j:j+4]
        cost=evaluate_cost(segment[:-1],segment[-1] ,current.MusicInput.tonality, mode="sum", first_on_beat=j%4)
        if cost>maxcost:
            maxxost, destroy_idx=cost , j
    for i in range(1,5):
        for j in range(destroy_idx, destroy_idx+4 ):
            destroyed[i,j] =-100     
    ##https://github.com/N-Wouda/ALNS/blob/master/examples/travelling_salesman_problem.ipynb
    
    return destroyed   
    
def destroy_2(current, random_state): ## worst scritical-cost removal
    destroyed = current.copy()  
    return destroyed
        
### Repair operators ###
# You can follow the example and implement repair_2, repair_3, etc

def repair_1(destroyed, random_state):#greedy_repair
    mp_model = MPModel("test", music, destroyed ,chord_vocab,
                        #hard_constraints, 
                        soft_constraint_w_weights, 
                        file_progression_cost=file_progression_cost)
    repaired = mp_model.solve()
    return repaired
#%%
if __name__ == '__main__':
    ini=solution.copy()
    # construct random initialized solution
    harmony=Harmony(music, ini)
    print("Initial solution objective is {}.".format(harmony.objective()))
  

    # ALNS
    random_state = rnd.RandomState(606)
    alns = ALNS(random_state)
    # add destroy
    # You should add all your destroy and repair operators
    
    alns.add_destroy_operator(destroy_1)
    #alns.add_destroy_operator(destroy_2)
    
    # add repair
    alns.add_repair_operator(repair_1)
    
    # run ALNS
    # select cirterion
    criterion = HillClimbing()
    # assigning weights to methods
    omegas =  [3, 2, 1, 0.5]
    lambda_ = 0.8
    result = alns.iterate(solution, omegas, lambda_, criterion,
                          iterations=1000, collect_stats=True)
    # result
    solution = result.best_state
    
    
    df_soluiton=pd.DataFrame(np.array(solution))
    df_soluiton.to_csv("ALNS_start.csv", index=False, header=False)
    
#%%    
    
    objective = solution.objective()
    print('Best heuristic objective is {}.'.format(objective))
    
    
    # visualize final solution and gernate output file
    save_output('HE CHEN', solution, 'solution')
