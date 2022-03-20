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
    test_function(f)