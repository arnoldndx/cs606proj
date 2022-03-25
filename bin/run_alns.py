'''
Description: ALNS

'''

#import argparse
import numpy as np
import numpy.random as rnd
import pandas as pd

import sys

sys.path.append('../')

from src.ALNS.alns import ALNS
from src.ALNS.alns.criteria import HillClimbing, SimulatedAnnealing, RecordToRecordTravel

from src.chord import Chord
from src.musical_work_input import MusicalWorkInput,Harmony

from src.mp_model_for_ALNS_construction import MPModel
import src.evaluate_v0
from src.midi_processing import *   
# Importing Musical Corpus
musical_work_df = pd.read_csv("../data/sample_input.csv")
musical_corpus = []
for i, title, meter, key, tonality, first_on_beat, melody in musical_work_df.itertuples():
    musical_corpus.append(MusicalWorkInput(title, meter, key, tonality, first_on_beat, [int(x) for x in melody.split(',')]))

# =============================================================================
# 0 Ach bleib' bei unsm Herr Jesu Christ
# 1 Ach Gott, erhor', mein Seufzen und Wehklagen
# 2 Ach Gott und Herr, wie gross und schwer
# 3 Ode to Joy
# 4 Old 124th
# 5 Ach, was soll ich Sunder machen

# =============================================================================
import timeit
start = timeit.default_timer()
music=musical_corpus[-1]
print(music.title, music.key, music.tonality, 
      music.first_on_beat,music.melody, music.reference_note)



# Defining dictionary of hard and soft constraint options:
# hard_constraint_options = ['musical input', 'voice range', 'chord membership', 'first last chords',
#                            'chord bass repetition', 'adjacent bar chords', 'voice crossing', 'parallel movement',
#                           'chord spacing']

# soft_constraint_options = ['chord progression', 'chord bass repetition', 'leap resolution',
#                            'melodic movement', 'note repetition', 'parallel movement', 'voice overlap', 'adjacent bar chords',
#                            'chord spacing', 'distinct notes', 'voice crossing', 'voice range']

hard_constraint_options = ['musical input', 'voice range', 'chord membership', 'first last chords',
                           'chord repetition', 'chord bass repetition', 'adjacent bar chords', 'voice crossing', 'parallel movement',
                          'chord spacing', 'incomplete chord', 'chord distribution']
hard_constraints = {x: True if x in ['musical input', 'voice range', 'chord membership', 'first last chords', 'voice crossing', 'parallel movement', 'chord spacing', 'incomplete chord']
                    else False for x in hard_constraint_options}

soft_constraint_options = ['chord progression', 'chord repetition', 'chord bass repetition', 'leap resolution',
                           'melodic movement', 'note repetition', 'parallel movement', 'voice overlap', 'adjacent bar chords',
                           'chord spacing', 'distinct notes', 'incomplete chord', 'voice crossing', 'voice range',
                           'second inversion', 'chord distribution']
# Model
# Model
#cp_model = CPModel("test", musical_corpus[0], chord_vocab)

# Importing Chord Vocabulary
if music.tonality=="major":
    chord_df = pd.read_csv("../data/chord_vocabulary_major.csv", index_col = 0)
else:
    chord_df = pd.read_csv("../data/chord_vocabulary_minor.csv", index_col = 0)
chord_vocab = []
for index, name, note_intervals in chord_df.itertuples():
#for name, note_intervals in chord_df.itertuples():
    chord_vocab.append(Chord(index, name, set(int(x) for x in note_intervals.split(','))))
    
# Defining dictionary of weights for each soft constraint options:
weight_df = pd.read_csv("../data/soft_constraint_weights.csv")
soft_constraint_w_weights={}
for _,name, w in weight_df.itertuples(): #name population is same as soft_constraint_options
    soft_constraint_w_weights[name]=float(w)
file_progression_cost="chord_progression_major_v1.csv" if music.tonality=="major" else "chord_progression_minor_v1.csv"
dic_bestchord_fwd=src.music_functions.func_get_best_progression_chord(file_progression_cost, "fwd")
dic_bestchord_bwd=src.music_functions.func_get_best_progression_chord(file_progression_cost, "bwd")


# Model
mp_model = MPModel("test", music,[], chord_vocab,
                    hard_constraints, 
                    soft_constraint_w_weights, 
                    file_progression_cost=file_progression_cost
                    ,timelimit=60)
solution = mp_model.solve()


df_soluiton=pd.DataFrame(np.array(solution))
df_soluiton.to_csv("ALNS_start.csv", index=False, header=False)


cost=src.evaluate_v0.evaluate_cost(solution[:-1],solution[-1] , tonality=music.tonality, mode="L")   
print(cost)
#%%
def destroy_1(current, random_state): ## greedy worst sum-of-cost removal of 2nd, 3rd or 4th note 

    #class Harmony(State)
    
    destroyed = current.copy()   
    maxcost, destroy_idx=0,0
    for j in range(0, current.N-1, 4):
        segment=  [current.HarmonyInput[k][j:j+5] for k in range(5)] #including 1st note of next bar for evaluation
      
        cost=src.evaluate_v0.evaluate_cost(segment[:-1],segment[-1] ,
                                        current.MusicInput.tonality, 
                                        mode="sum", 
                                        first_on_beat=1)
        if cost>maxcost:
            maxcost, destroy_idx=cost , j
    for i in range(1,5):
        for j in range(destroy_idx+1, destroy_idx+4 ):
            destroyed.HarmonyInput[i][j] =-100     
    
    return destroyed   
    
def destroy_2(current, random_state): ## random removal of whole bar
    destroyed = current.copy()   
    destroy_idx=4* rnd.randint(0,current.N //4)

    for i in range(1,5):
        for j in range(destroy_idx, destroy_idx+4 ):
            destroyed.HarmonyInput[i][j] =-100     
    ##https://github.com/N-Wouda/ALNS/blob/master/examples/travelling_salesman_problem.ipynb
    
    return destroyed   
        
### Repair operators ###



def repair_1(destroyed, random_state):#greedy_repair, fix 2 missing chords in a row
    repaired=destroyed.copy()
    length=len(repaired.HarmonyInput[-1])
    
    #repaired.HarmonyInput[-1]==-100
    counter=0
    for j in range(1,length):
        if  repaired.HarmonyInput[-1][j]==-100 and counter<=1:
            repaired.HarmonyInput[-1][j]=dic_bestchord_fwd.get(repaired.HarmonyInput[-1][j-1],0)
            counter+=1
        elif repaired.HarmonyInput[-1][j]!=-100:
            counter=0
    
    mp_model = MPModel("test", music, destroyed.HarmonyInput ,chord_vocab,
                        hard_constraints, 
                        soft_constraint_w_weights, 
                        file_progression_cost=file_progression_cost,
                        timelimit=30)
    solved= mp_model.solve()
    repaired.HarmonyInput = solved
    repaired.HarmonyOutput = solved
    repaired.notes = solved[:-1]
    repaired.chords = solved[-1]
    
    return repaired
def repair_2(destroyed, random_state):#greedy_repair, fix 2 missing chords in a row
    repaired=destroyed.copy()
    length=len(repaired.HarmonyInput[-1])
    
    #repaired.HarmonyInput[-1]==-100
    counter=0
    for j in range(length-2,0,-1):
        if  repaired.HarmonyInput[-1][j]==-100 and counter<=1:
            repaired.HarmonyInput[-1][j]=dic_bestchord_bwd.get(repaired.HarmonyInput[-1][j+1],0)
            counter+=1
        elif repaired.HarmonyInput[-1][j]!=-100:
            counter=0
    
    mp_model = MPModel("test", music, destroyed.HarmonyInput ,chord_vocab,
                        hard_constraints, 
                        soft_constraint_w_weights, 
                        file_progression_cost=file_progression_cost,
                        timelimit=30)
    solved= mp_model.solve()
    repaired.HarmonyInput = solved
    repaired.HarmonyOutput = solved
    repaired.notes = solved[:-1]
    repaired.chords = solved[-1]
    
    return repaired
def alns_export_midi(notes, instruments = [20]*4, beat = 500, filepath = '../outputs'):
    array_to_midi(notes, instruments, beat,
                  dest_file_path = '{}/cp_{}_{}_{}_{}.mid'.format(
                      filepath, self.name, self.musical_input.title, self.hard_constraint_encoding, self.soft_constraint_encoding))

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
    
    alns.add_destroy_operator(destroy_1)
    alns.add_destroy_operator(destroy_2)
    
    # add repair
    alns.add_repair_operator(repair_1)
    alns.add_repair_operator(repair_2)
    # run ALNS
    # select cirterion
    criterion = HillClimbing()
    # assigning weights to methods
    omegas =  [3, 2, 1, 0.5]
    lambda_ = 0.8

    result = alns.iterate(harmony, omegas, lambda_, criterion,
                          iterations=90, collect_stats=True)
    # result
    ALNS_solution = result.best_state   
    
    df_soluiton=pd.DataFrame(np.array(ALNS_solution.HarmonyInput))
    df_soluiton.to_csv("../outputs/ALNS_end.csv", index=False, header=False)
    
#%%    
    
    objective = ALNS_solution.objective()
    print('Best heuristic objective is {}.'.format(objective))
    filepath = '../outputs'
    hard_constraint_encoding, soft_constraint_encoding = src.music_functions.encode_constraints(hard_constraints, soft_constraint_w_weights)
    dest_file_path = '{}/alns_{}_{}_{}_{}.mid'.format(filepath, music.title, music.tonality, hard_constraint_encoding, soft_constraint_encoding)
     
    array_to_midi(ALNS_solution.HarmonyInput[:4], [53]*4, 600,dest_file_path )   
    
    stop = timeit.default_timer()

    print('ALNS Run Time: ', stop - start)  
    
