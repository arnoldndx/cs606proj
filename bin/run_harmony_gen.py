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
from docplex.cp.model import CpoModel

#Custom Imports
sys.path.append('../')
from src.chord import Chord
from src.musical_work_input import MusicalWorkInput
from src.cp_model import CPModel

# Importing Chord Vocabulary
chord_df = pd.read_csv("../data/chord_vocabulary.csv", index_col = 0)
chord_vocab = []
for name, note_intervals in chord_df.itertuples():
    chord_vocab.append(Chord(name, set(int(x) for x in note_intervals.split(','))))

# Importing Musical Corpus
musical_work_df = pd.read_csv("../data/sample_input.csv")
musical_corpus = []
for i, title, meter, key, tonality, first_on_beat, melody in musical_work_df.itertuples():
    musical_corpus.append(MusicalWorkInput(title, meter, key, tonality, first_on_beat, [int(x) for x in melody.split(',')]))

# Model
cp_model = CPModel("test", musical_corpus[0], chord_vocab)
cp_model.define_decision_variables()
cp_model.hard_constraint_musical_input()
cp_model.hard_constraint_voice_ranges()
#cp_model.hard_constraint_chord_grades() #buggy, don't run this yet
cp_model.hard_constraint_first_last_chords()
cp_model.hard_constraint_adjacent_bar_chords()
cp_model.hard_constraint_voice_crossing()
solution = cp_model.solve()
