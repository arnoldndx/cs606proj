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
from docplex.cp.model import CpoModel

#Custom Imports
sys.path.append('../')
from src.chord import Chord
from src.musical_work_input import MusicalWorkInput
#from src.cp_model import CPModel
from src.mp_model import MPModel

# Importing Chord Vocabulary
chord_df = pd.read_csv("../data/chord_vocabulary_major.csv", index_col = 0)
chord_vocab = []
for name, note_intervals in chord_df.itertuples():
    chord_vocab.append(Chord(name, set(int(x) for x in note_intervals.split(','))))

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

print(musical_corpus[-1].title, musical_corpus[-1].key, musical_corpus[-1].tonality, 
      musical_corpus[-1].first_on_beat,musical_corpus[-1].melody, musical_corpus[-1].reference_note)

#%%
mp_model = MPModel("test", musical_corpus[0], chord_vocab,
                    #hard_constraints, 
                    #soft_constraints, 
                    file_progression_cost="chord_progression_major_v1.csv")
solution = mp_model.solve()
#solution = cp_model.solve()
