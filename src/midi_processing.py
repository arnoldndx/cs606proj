# -*- coding: utf-8 -*-
'''
Created on Mon Feb 28 20:17:15 2022

These functions and classes are for midi processing

Dummy code to define classes and functions from the assignment 2 script. Start with this first.

'''
import os
import sys

import pretty_midi
import pandas as pd

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

fn = os.path.join(base_path, 'data', 'Bach Chorales', '01AusmeinesHerz.mid')
midi_data = pretty_midi.PrettyMIDI(fn)
midi_list = []

print(midi_data.get_tempo_changes())

for instrument in midi_data.instruments:
    for note in instrument.notes:
        start = note.start
        end = note.end
        pitch = note.pitch
        velocity = note.velocity
        midi_list.append([start, end, pitch, velocity, instrument.name])
        
midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))

df = pd.DataFrame(midi_list, columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument'])

print(df)