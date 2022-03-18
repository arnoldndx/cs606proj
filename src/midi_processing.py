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


def midi_to_array(midi_file):
    '''
    Function to convert midi file to array for processing.
    
    Parameters
    ----------
    midi_file : str
        File path of the midi file

    Returns
    -------
    midi_array : list
        Array of notes (int) standardised to a constant beat

    '''
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    midi_list = []
    
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch
            velocity = note.velocity
            midi_list.append([start, end, pitch, velocity, instrument.name])
            
    midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))
    
    # generate array by comparing the timings and standardising the beat
    midi_array = midi_list
    
    #df = pd.DataFrame(midi_list, columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument'])

    return midi_array

def array_to_midi(midi_array, instruments, beat, dest_file_path = '../outputs/model_output.mid'):
    '''
    Function to convert array to midi file

    Parameters
    ----------
    midi_array : list of int (0-127)
        Array of notes (int) standardised to a constant beat. Find the note numbers mapped at: https://www.music.mcgill.ca/~ich/classes/mumt306/StandardMIDIfileformat.html#BMA1_3
        
    instruments : list of int (0-127)
        Array of four program numbers of the selected instrument for each voice. Find the patch map at: https://www.music.mcgill.ca/~ich/classes/mumt306/StandardMIDIfileformat.html#BMA1_4        
        
    beat : int
        An integer representing the time length of each note in microseconds
        
    file_name : str
        Name of output file

    Returns
    -------
        Writes the midi file to output

    '''
    # Check validity of instrument array
    if len(instruments) != 4:
        raise Exception('Error, length of instrument array should be 4')
    
    # Create a PrettyMIDI object to store the compiled music, tempo in bpm converted from beat
    midi_output = pretty_midi.PrettyMIDI(tempo = 60000/beat)
        
        
    for i in range(len(midi_array)):
        # Create a PrettyMIDI instrument with chosen instrument
        instrument = pretty_midi.Instrument(program=instruments[i])
        # Iterate over note names, which will be converted to note number later
        for note_number in midi_array[i]:
            # Create a Note instance, starting at 0 and ending at 1 beat (in s)
            note = pretty_midi.Note(
                velocity=100, pitch=note_number, start=0, end=beat*1000)
            # Add it to our cello instrument
            instrument.notes.append(note)
        # Add the instrument to the PrettyMIDI object
        midi_output.instruments.append(instrument)
        
    # Write out the MIDI data
    midi_output.write(dest_file_path)
    
    return