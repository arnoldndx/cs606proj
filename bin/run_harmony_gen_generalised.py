# -*- coding: utf-8 -*-
'''
Created on Mon Feb 28 20:17:15 2022
Edited by Gab on Thu Mar 3 19:26:00 2022
'''
#Standard Imports
import os
import sys
import pandas as pd
# from collections import defaultdict
import timeit
import argparse
import logging

# for MP/CP model
from docplex.cp.model import CpoModel

# for ALNS
from src.ALNS.alns import ALNS
from src.ALNS.alns.criteria import HillClimbing, SimulatedAnnealing, RecordToRecordTravel
import src.evaluate

#Custom Imports
sys.path.append('../')
from src.chord import Chord
from src.musical_work_input import MusicalWorkInput
#from src.cp_model import CPModel
from src.mp_model import MPModel
from src.midi_processing import *
from src.train_weights import *

logger = logging.getLogger(__file__)

# Setting up arguments
parser.add_argument('--method', type = str, default = 'mp', choices=['mp', 'cp', 'ga', 'alns'])
parser.add_argument("--file", type = str, default = 'harmony_gen', help = "Filename prefix. "
                                                                         "You should give a meaningful name for easy tracking.")
parser.add_argument('--weights', type = str, default = 'defined', choices=['defined', 'trained'])
parser.add_argument('--weights_data', type = str, help = 'Filepath for weights training data. Should be either a csv file for defined weights,'
                                                        'or a folder containing midi files for trained wweights. Data should be in ../data/')
parser.add_argument('--input_melody', type = str, default = '../data/test_melody.mid', help = "Filepath for the input melody. Valid filetypes: .csv, .mid")

#%%
# Starting up
args = parser.parse_args()
logger.info(f'==============================================')
logger.info('Importing melody')

# Importing melody
if args.input[-3:] == 'csv':
    musical_work_df = pd.read_csv(args.input)
    musical_corpus = []
    for i, title, meter, key, tonality, first_on_beat, melody in musical_work_df.itertuples():
        musical_corpus.append(MusicalWorkInput(title, meter, key, tonality, first_on_beat, [int(x) for x in melody.split(',')]))
elif args.input[-3:] == 'mid':
    musical_work_df = midi_to_array(args.input)
    input_midi = args.input_melody
    melody, tempo_interval, meter, key, tonality, first_on_beat = midi_to_array()
    musical_corpus = []
    musical_corpus.append(MusicalWorkInput(os.path.split(args.input)[1][:-4], meter, key, tonality, first_on_beat, melody))
 
#%% 
# Defining dictionary of hard and soft constraint options:
hard_constraint_options = ['musical input', 'voice range', 'chord membership', 'first last chords',
                           'chord repetition', 'chord bass repetition', 'adjacent bar chords', 'voice crossing', 'parallel movement',
                          'chord spacing']

soft_constraint_options = ['chord progression', 'chord repetition', 'chord bass repetition', 'leap resolution',
                           'melodic movement', 'note repetition', 'parallel movement', 'voice overlap', 'adjacent bar chords',
                           'chord spacing', 'distinct notes', 'voice crossing', 'voice range']

#%%
logger.info(f'==============================================')
logger.info(f'Preparing weights for soft constraints')
# define weights
if args.weights == 'defined':
    weight_df = pd.read_csv("../data/soft_constraint_weights.csv")
# elif args.weights == 'trained':
    # weight_df = train_weights()    
    
#%%
logger.info(f'==============================================')
logger.info(f'Generating Harmony with {args.method} model')
if args.method == 'mp':
    #%%
    # Model
    #cp_model = CPModel("test", musical_corpus[0], chord_vocab)
    music=musical_corpus[1]
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
        
    # SHOULD SET AN ARGUMENT FOR WEIGHTS SETTING, EITHER READ FROM DICTIONARY OR GET FROM GRADIENT DESCENT
    # Defining dictionary of weights for each soft constraint options:
    soft_constraint_w_weights={}
    for _,name, w in weight_df.itertuples(): #name population is same as soft_constraint_options
        soft_constraint_w_weights[name]=float(w)
    file_progression_cost="chord_progression_major_v1.csv" if music.tonality=="major" else "chord_progression_minor_v1.csv"
    # Model
    mp_model = MPModel("test", music, chord_vocab,
                        #hard_constraints, 
                        soft_constraint_w_weights, 
                        file_progression_cost=file_progression_cost)
    solution, midi_array = mp_model.solve()
#%%
elif args.method == 'cp':
    

#%%    
elif args.method == 'alns':
    #%%    
    def destroy_1(current, random_state): ## greedy worst sum-of-cost removal of 2nd, 3rd or 4th note 

        #class Harmony(State)
        
        destroyed = current.copy()   
        maxcost, destroy_idx=0,0
        for j in range(0, current.N-1, 4):
            segment=  [current.HarmonyInput[k][j:j+5] for k in range(5)] #including 1st note of next bar for evaluation
          
            cost=src.evaluate.evaluate_cost(segment[:-1],segment[-1] ,
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
    ini=solution.copy()
    # construct random initialized solution
    harmony=Harmony(music, ini)
    print("ALNS - Initial solution objective is {}.".format(harmony.objective()))
  
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
                          iterations=60, collect_stats=True)
    # result
    ALNS_solution = result.best_state   
    
    df_soluiton=pd.DataFrame(np.array(ALNS_solution.HarmonyInput))
    df_soluiton.to_csv("../outputs/ALNS_end.csv", index=False, header=False)  
    
    print('Best heuristic objective is {}.'.format(ALNS_solution.objective()))
    
    # extract solution
    midi_array = ALNS_solution.HarmonyInput[:4]
    
# 
stop = timeit.default_timer()

print('ALNS Run Time: ', stop - start)  

# generate the solution as a midi file
logger.info(f'==============================================')
logger.info(f'Output melody written to: {results_path}')

# generate the solution as a midi file
array_to_midi(midi_array, [53]*4, 600)

#%%
# for music in musical_corpus:
#     print(music.title, music.key, music.tonality, 
#           music.first_on_beat,music.melody, music.reference_note)
# # Importing Chord Vocabulary
#     if music.tonality=="major":
#         chord_df = pd.read_csv("../data/chord_vocabulary_major.csv", index_col = 0)
#     else:
#         chord_df = pd.read_csv("../data/chord_vocabulary_minor.csv", index_col = 0)
#     chord_vocab = []
#     for name, note_intervals in chord_df.itertuples():
#         chord_vocab.append(Chord(name, set(int(x) for x in note_intervals.split(','))))
        
#     # Defining dictionary of weights for each soft constraint options:
#     weight_df = pd.read_csv("../data/soft_constraint_weights.csv")
#     soft_constraint_w_weights={}
#     for _,name, w in weight_df.itertuples(): #name population is same as soft_constraint_options
#         soft_constraint_w_weights[name]=float(w)
#     file_progression_cost="chord_progression_major_v1.csv" if music.tonality=="major" else "chord_progression_minor_v1.csv"
#     # Model
#     mp_model = MPModel("test", music, chord_vocab,
#                         #hard_constraints, 
#                         soft_constraint_w_weights, 
#                         file_progression_cost=file_progression_cost)
#     solution = mp_model.solve()
