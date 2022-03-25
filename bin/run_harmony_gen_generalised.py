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
import copy

sys.path.append('../')

# Custom Imports
from src.chord import Chord
from src.musical_work_input import MusicalWorkInput
from src.midi_processing import *
# from src.learning_weights import *

# for MP/CP model
from docplex.cp.model import CpoModel
from src.mp_model import MPModel
from src.cp_model import CPModel

# for ALNS
from src.ALNS.alns import ALNS
from src.ALNS.alns.criteria import HillClimbing, SimulatedAnnealing, RecordToRecordTravel
import src.evaluate_v0
from src.mp_model_for_ALNS_construction import MPModelALNS
import src.evaluate_v0


logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


# Setting up arguments
parser = argparse.ArgumentParser()
parser.add_argument('--method', type = str, default = 'mp', choices=['mp', 'cp', 'ga', 'alns'])
parser.add_argument("--file", type = str, default = 'harmony_gen', help = "Filename prefix. "
                                                                         "You should give a meaningful name for easy tracking.")
parser.add_argument('--weights', type = str, default = 'defined', choices=['defined', 'trained'])
parser.add_argument('--weights_data', type = str, default = "../data/soft_constraint_weights_temp.csv",  help = 'Filepath for weights training data. Should be either a csv file for defined weights,'
                                                        'or a folder containing midi files for trained wweights. Data should be in ../data/')
parser.add_argument('--hard_constraints_choice', type = str, default = '../data/hard_constraint_choice.csv', help = 'Filepath for hard constraint choices')
parser.add_argument('--time_limit', type = int, default = 600, help = 'Time limit for iterations (MP/CP) or Iteration limit for ALNS')
parser.add_argument('--input_melody', type = str, default = '../data/test_melody.mid', help = "Filepath for the input melody. Valid filetypes: .csv, .mid")

#%%
# Starting up
args = parser.parse_args()
logger.info(f'==============================================')
logger.info('Importing melody')

# Importing melody
if args.input_melody[-3:] == 'csv':
    logger.info('Melody is .csv')
    musical_work_df = pd.read_csv(args.input_melody)
    musical_corpus = []
    for i, title, meter, key, tonality, first_on_beat, melody in musical_work_df.itertuples():
        musical_corpus.append(MusicalWorkInput(title, meter, key, tonality, first_on_beat, [int(x) for x in melody.split(',')]))
elif args.input_melody[-3:] == 'mid':
    logger.info('Melody is .mid')
    input_midi = args.input_melody
    #melody, tempo_interval, meter, key, tonality, first_on_beat = midi_to_array(input_midi)
    melody, tempo_interval = midi_to_array_quick(input_midi)
    filename = str.split(str.split(args.input_melody,'/')[-1],'_')
    first_on_beat = int(str.split(filename[-1],'.')[0])
    tonality = filename[-2]
    key = int(filename[-3])
    meter = int(filename[-4])
    song_title = '_'.join(filename[2:-4])
    musical_corpus = []
    musical_corpus.append(MusicalWorkInput(song_title, meter, key, tonality, first_on_beat, list(filter(None,melody[0]))))
else:
    print(args.input_melody[-3])
    raise Exception('Error: Melody not loaded')
 
#%% 
# Defining dictionary of hard and soft constraint options:
hard_constraint_options = ['musical input', 'voice range', 'chord membership', 'first last chords',
                           'chord repetition', 'chord bass repetition', 'adjacent bar chords', 'voice crossing', 'parallel movement',
                          'chord spacing', 'incomplete chord', 'chord distribution']

soft_constraint_options = ['chord progression', 'chord repetition', 'chord bass repetition', 'leap resolution',
                           'melodic movement', 'note repetition', 'parallel movement', 'voice overlap', 'adjacent bar chords',
                           'chord spacing', 'distinct notes', 'incomplete chord', 'voice crossing', 'voice range',
                           'second inversion', 'chord distribution']

# choosing hard constraints
hard_constraint_choices = list(pd.read_csv(args.hard_constraints_choice).columns)
hard_constraints = {x: True if x in hard_constraint_choices
                    else False for x in hard_constraint_options}

#%%
logger.info(f'==============================================')
logger.info(f'Preparing weights for soft constraints')
# define weights
if args.weights == 'defined':
    weight_df = pd.read_csv(args.weights_data)
# elif args.weights == 'trained':
    # weight_df = train_weights()    
    

# Defining dictionary of weights for each soft constraint option:
soft_constraint_w_weights = {}
for _, name, w in weight_df.itertuples(): #name population is same as soft_constraint_options
    soft_constraint_w_weights[name] = float(w)
    
assert sum(v for v in soft_constraint_w_weights.values() if v > 0) == 100

#%%
logger.info(f'==============================================')
logger.info(f'Generating Harmony with {args.method} model')

# get start time
start = timeit.default_timer()

music = musical_corpus[-1]
logger.info(f'Title: {music.title}, Key: {music.key}, Tonality: {music.tonality}, Onset: {music.first_on_beat}, Melody: {music.melody}, Ref #C: {music.reference_note}')

if args.method == 'mp':
    #%%
    # Model
    #cp_model = CPModel("test", musical_corpus[0], chord_vocab)
    
    # Importing Chord Vocabulary
    if music.tonality=="major":
        chord_df = pd.read_csv("../data/chord_vocabulary_major.csv", index_col = 0)
    else:
        chord_df = pd.read_csv("../data/chord_vocabulary_minor.csv", index_col = 0)
    chord_vocab = []
    for index, name, note_intervals in chord_df.itertuples():
        chord_vocab.append(Chord(index, name, set(int(x) for x in note_intervals.split(','))))
        
        
    file_progression_cost="../data/chord_progression_major_v1.csv" if music.tonality=="major" else "../data/chord_progression_minor_v1.csv"
    # Model
    mp_model = MPModel("test", music, chord_vocab,
                        #hard_constraints, 
                        soft_constraint_w_weights, 
                        file_progression_cost=file_progression_cost,
                        timelimit=args.time_limit)
        
    midi_array_with_chords = mp_model.solve()
    
    # generate df_solution for csv solution gen
    df_solution = pd.DataFrame(np.array(midi_array_with_chords))
    
    # generate midi_array for midi gen
    midi_array = midi_array_with_chords[:4]
        
    
#%%
elif args.method == 'cp':
    # Importing Chord Vocabulary
    chord_df_major = pd.read_csv("../data/chord_vocabulary_major.csv", index_col = 0)
    chord_df_minor = pd.read_csv("../data/chord_vocabulary_minor.csv", index_col = 0)
    chord_vocab_major, chord_vocab_minor = [], []
    for index, name, note_intervals in chord_df_major.itertuples():
        chord_vocab_major.append(Chord(index, name, [int(x) for x in note_intervals.split(',')]))
    for index, name, note_intervals in chord_df_minor.itertuples():
        chord_vocab_minor.append(Chord(index, name, [int(x) for x in note_intervals.split(',')]))
    
    # Defining penalties for chord progression
    penalties_chord_progression_major = pd.read_csv("../data/chord_progression_major.csv", header = 1, index_col = 0)
    penalties_chord_progression_minor = pd.read_csv("../data/chord_progression_minor.csv", header = 1, index_col = 0)
    penalties_chord_progression_major = dict(penalties_chord_progression_major.stack())
    penalties_chord_progression_minor = dict(penalties_chord_progression_minor.stack())
        
    
    # Defining which hard constraints to use
    hard_constraints = {x: True if x in ['musical input', 'voice range', 'chord membership', 'first last chords',
                                         'voice crossing', 'parallel movement',
                                         'chord spacing', 'incomplete chord'] else False for x in hard_constraint_options}
    
    # Model
    
    if music.tonality == 'major':
        penalties_chord_progression = penalties_chord_progression_major
        chord_vocab = chord_vocab_major
    else:
        penalties_chord_progression = penalties_chord_progression_minor
        chord_vocab = chord_vocab_minor        
    
    #Defining Model
    cp_model = CPModel('Completed', music, chord_vocab, penalties_chord_progression,
                       hard_constraints,
                       soft_constraint_w_weights)
    
    #Solving Model
    solution = cp_model.solve(log_output = True, TimeLimit = args.time_limit, LogVerbosity = 'Verbose')
    result = cp_model.get_solution()
          
    # generate df_solution for csv solution gen
    solved_harmony = copy.deepcopy(result['Notes'])
    solved_harmony.append(result['Chords'])
                                            
    df_solution = pd.DataFrame(np.array(solved_harmony))
    
    # generate midi_array for midi gen
    midi_array = result['Notes']
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
    # Importing Chord Vocabulary
    if music.tonality=="major":
        chord_df = pd.read_csv("../data/chord_vocabulary_major.csv", index_col = 0)
    else:
        chord_df = pd.read_csv("../data/chord_vocabulary_minor.csv", index_col = 0)
    chord_vocab = []
    for index, name, note_intervals in chord_df.itertuples():
    #for name, note_intervals in chord_df.itertuples():
        chord_vocab.append(Chord(index, name, set(int(x) for x in note_intervals.split(','))))
        
 
    file_progression_cost = "chord_progression_major_v1.csv" if music.tonality == "major" else "chord_progression_minor_v1.csv"
    dic_bestchord_fwd=src.music_functions.func_get_best_progression_chord(file_progression_cost, "fwd")
    dic_bestchord_bwd=src.music_functions.func_get_best_progression_chord(file_progression_cost, "bwd")


    # Construction heuristic (MP model)
    mp_model = MPModel("test", music, [], chord_vocab,
                        hard_constraints, 
                        soft_constraint_w_weights, 
                        file_progression_cost = file_progression_cost,
                        timelimit=60)
    
    
    solution = mp_model.solve()

    
    ini = solution.copy()
    # construct random initialized solution
    harmony = Harmony(music, ini)
    logger.info(f'ALNS - Initial solution objective is {harmony.objective()}')
  
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
                          iterations=args.time_limit, collect_stats=True)
    # result
    ALNS_solution = result.best_state   
    
    df_solution = pd.DataFrame(np.array(ALNS_solution.HarmonyInput))
    
    logger.info(f'Best heuristic objective is {ALNS_solution.objective()}')
    
    # extract solution
    midi_array = ALNS_solution.HarmonyInput[:4]
    
# Get the stop time
stop = timeit.default_timer()

run_time = stop - start

# print run time
logger.info(f'==============================================')
logger.info(f'Run Time: {run_time}')  

#%%
# define the destination and name of file
filepath = '../outputs'
hard_constraint_encoding, soft_constraint_encoding = src.music_functions.encode_constraints(hard_constraints, soft_constraint_w_weights)
dest_file_path = '{}/{}_{}_{}_{}_{}.mid'.format(filepath, 
                                                args.method, 
                                                music.title, 
                                                music.tonality, 
                                                hard_constraint_encoding, 
                                                soft_constraint_encoding)

# generate the solution as a midi file
logger.info(f'==============================================')
logger.info(f'Writing outputs...')

# generate the solution as a midi file
array_to_midi(midi_array, [53]*4, 600, dest_file_path=dest_file_path)

# generate solution as a csv
df_solution.to_csv('{}/{}_{}_{}_{}_{}.csv'.format(filepath, 
                                                  args.method, 
                                                  music.title, 
                                                  music.tonality, 
                                                  hard_constraint_encoding, 
                                                  soft_constraint_encoding), index=False, header=False)

logger.info(f'==============================================')
logger.info(f'Output melody written to: {dest_file_path}')
