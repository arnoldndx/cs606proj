# -*- coding: utf-8 -*-
'''
Created on Mon Feb 28 20:17:15 2022
Edited by Gab on Thu Mar 3 19:26:00 2022

[NOTES]

CONVENTIONS

Middle C is 24

Decision variables are expressed as a dictionary
x = {S:[s1,s2,...sj],
     A:[a1,a2,...aj],
     T:[t1,t2,...tj],
     B:[b1,b2,...bj]}

c = [c1,c2,...cj] #records chords at each timestamp for computation of soft constraints

GIVEN DATA

#List of chords (based on a 12 note scale, can convert to note integers later)
chord_list = [[0,3,7],   #i
              [0,4,7],   #I
              ...
              ...
              ]

#Voice ranges


#Key shift
key_shift = 0

#Tonality
tonality = 'major' or 'minor'

HARD CONSTRAINTS

1.  x[S,j] = sample_input[j] #melody is given, the rest of the composition must match
2.  lb[i] <= x[i][j] <= ub[i] #each voice must be within its range, i in [S,A,T,B]
3.  if tonality == 'major':
        x[][0] and x[][n] must match chord I
    elif tonality == 'minor':
        x[][0] must match chord i and x[][n] must match chord i or I

SOFT CONSTRAINTS
1.  

'''
#Standard Imports

import os
import sys
import pandas as pd
# from collections import defaultdict
import timeit


#Custom Imports
sys.path.append('../')
from src.chord import Chord
from src.musical_work_input import MusicalWorkInput
#from src.cp_model import CPModel
from src.mp_model import MPModel

from src.midi_processing import *
from src.music_functions import encode_constraints
import timeit
start = timeit.default_timer()
# Importing Musical Corpus
musical_work_df = pd.read_csv("../data/sample_input.csv")
musical_corpus = []
for i, title, meter, key, tonality, first_on_beat, melody in musical_work_df.itertuples():
    musical_corpus.append(MusicalWorkInput(title, meter, key, tonality, first_on_beat, [int(x) for x in melody.split(',')]))

# Defining dictionary of hard and soft constraint options:
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
music=musical_corpus[-1]
print(music.title, music.key, music.tonality, 
      music.first_on_beat,music.melody, music.reference_note)
# Importing Chord Vocabulary !!! input had been changed 
if music.tonality=="major":
    chord_df = pd.read_csv("../data/chord_vocabulary_major.csv", index_col = 0)
else:
    chord_df = pd.read_csv("../data/chord_vocabulary_minor.csv", index_col = 0)
chord_vocab = []
for _,name, note_intervals in chord_df.itertuples():
    chord_vocab.append(Chord(name, set(int(x) for x in note_intervals.split(','))))
    
# Defining dictionary of weights for each soft constraint options:
weight_df = pd.read_csv("../data/soft_constraint_weights.csv")
soft_constraint_w_weights={}
for _,name, w in weight_df.itertuples(): #name population is same as soft_constraint_options
    soft_constraint_w_weights[name]=float(w)
file_progression_cost="chord_progression_major_v1.csv" if music.tonality=="major" else "chord_progression_minor_v1.csv"
# Model
mp_model = MPModel("test", music, chord_vocab,
                    #hard_constraints, 
                    soft_constraint_w_weights, 
                    file_progression_cost=file_progression_cost,timelimit=600)
         


midi_array_with_chords = mp_model.solve()
# generate the solution as a midi file

filepath = '../outputs'
hard_constraint_encoding, soft_constraint_encoding = encode_constraints(hard_constraints, soft_constraint_w_weights)
dest_file_path = '{}/mp_{}_{}_{}_{}.mid'.format(filepath, music.title, music.tonality, hard_constraint_encoding, soft_constraint_encoding)
 
array_to_midi(midi_array_with_chords[:4], [53]*4, 600, dest_file_path )   
stop = timeit.default_timer()

print('MP Run Time: ', stop - start) 

#%%
# import pandas as pd
# df_soluiton=pd.DataFrame(np.array(midi_array_with_chords))
# df_soluiton.to_csv("MP.csv", index=False, header=False)

