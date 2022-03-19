# -*- coding: utf-8 -*-
'''
Created on Mon Feb 28 20:17:15 2022

These functions and classes are for midi processing

Dummy code to define classes and functions from the assignment 2 script. Start with this first.

'''
import os
import sys
import copy
from itertools import groupby

import pretty_midi

def all_equal(iterable):
    '''
    Utility function to check that all items in list are the same. For checking array lengths for each voice.
    
    Parameters
    ----------
    iterable : list
        List of the lengths of each voice (i.e. list of notes).

    Returns
    -------
        True if all lengths are the same, False otherwise

    '''
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

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
    interval_list = []
    notes_list = []
    instruments =[]
    
    for instrument in midi_data.instruments:
        instruments.append(instrument.program)
        notes = []
        intervals = []
        for note in instrument.notes:
            interval = round((note.end - note.start) * 1000) # calculate the interval for standardisation (int in ms)
            pitch = note.pitch
            intervals.append(interval)
            notes.append(pitch)
        
        # add the voice to the notes_list array
        notes_list.append(notes)
        
        # add the voice to the notes_list array
        interval_list.append(intervals)
                
    # find minimum interval
    min_interval = min(min(interval_list[i]) for i in range(4))
    
    # create midi_array as an array of notes with a constant beat
    midi_array = []
    
    for i in range(len(notes_list)):
        k = 0
        voice = []
        for j in range(len(notes_list[i])):
            length = interval_list[i][j] // min_interval
            for l in range(length):
                k += 1
                voice.append(notes_list[i][j])
        
        # add voice to midi array
        midi_array.append(voice)
    
    # check that all voices have same length
    lengths = []
    for voice in midi_array:
        lengths.append(len(voice))
    
    if not all_equal(lengths):
        raise Exception('Error in array output, not all voices have same length')
        
    return midi_array, min_interval

def array_to_midi(midi_array, instruments, beat, dest_file_path = '../outputs/model_output.mid'):
    '''
    Function to convert array to midi file

    Parameters
    ----------
    midi_array : nested list of int (0-127)
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
    midi_output = pretty_midi.PrettyMIDI(initial_tempo = round(60000/beat))
        
        
    for i in range(len(midi_array)):
        # Create a PrettyMIDI instrument with chosen instrument
        instrument = pretty_midi.Instrument(program=instruments[i])
        # restart the note interval
        time = 0
        # Iterate over note names, which will be converted to note number later
        for note_number in midi_array[i]:
            # Create a Note instance, starting at 0 and ending at 1 beat (in s)
            note = pretty_midi.Note(
                velocity=100, pitch=note_number, start = time, end = time + beat/1000)
            # Add it to our cello instrument
            instrument.notes.append(note)
            # extend the time
            time += beat/1000
        # Add the instrument to the PrettyMIDI object
        midi_output.instruments.append(instrument)
        
    # Write out the MIDI data
    midi_output.write(dest_file_path)
    
    return