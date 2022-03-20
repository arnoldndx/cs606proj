# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:40:46 2022

@author: Arnold
"""
import os
import sys

sys.path.append('../')
from src.midi_processing import *

directory = '../data/bach'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
       
    # test midi_import
    midi_array, tempo = midi_to_array(f)
    
    # test midi export
    array_to_midi(midi_array, [19] * 4, tempo, dest_file_path = '../outputs/test_' + str(filename))