# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:40:46 2022

@author: Arnold
"""
import os
import sys
import logging

sys.path.append('../')
from src.midi_processing import *

directory = '../data/bach'

logger = logging.getLogger(__file__)

for filename in os.listdir(directory):
    # logger.info(f'==============================================')
    # logger.info(f'{filename}')
    # print('==============================================')
    # print(filename)
    # print('==============================================')
    
    f = os.path.join(directory, filename)
       
    # test midi_import
    #midi_array, tempo = midi_to_array(f)
    test_function(f)
    
    # test midi export
    #array_to_midi(midi_array, [0] * 4, tempo * 1000, dest_file_path = '../outputs/test_' + str(filename))