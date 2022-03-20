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
import numpy as np

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

def test_function(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    print('time signature: ',midi_data.time_signature_changes)
    print('interval length: ',round(60000/midi_data.get_tempo_changes()[1][0]))
    print('beat locations: ',midi_data.get_beats())
    print('note onsets: ',midi_data.get_onsets())
    print('note starts (in ms): ',[[round(note.start * 1000) for note in instrument.notes] for instrument in midi_data.instruments])
    print('note starts (in beats): ',[[round(note.start/round(60/midi_data.get_tempo_changes()[1][0],3),4) for note in instrument.notes] for instrument in midi_data.instruments])
    return
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
    
    # get time interval (in microseconds)
    
    # confirm that there are no tempo changes in the music (i.e. tempo change only occurs once at start)
    if len(midi_data.get_tempo_changes()[0]) > 1:
        raise Exception(f'There are tempo changes in {midi_file}, do not use')
    
    # convert the tempo into a time interval (ms)
    tempo_interval = round(60000/midi_data.get_tempo_changes()[1][0])
    
    smallest_beat = 1
    
    # check if any note is ending or starting on a sub-beat
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            sub_beat = 2 ** round(np.log2(round(round(note.start * 1000) / tempo_interval, 3) % 1, 3)) # sub_beat should be in powers of 2
            # check if the note is starting on a beat location
            if  sub_beat > 0:
                smallest_beat = sub_beat
    
    # create a set of timesteps
    time_steps = [min(min(starts) for starts in start_list)]
    end = min(min(starts) for starts in start_list)
    while end < max(max(ends) for ends in end_list):
        end += min_interval
        time_steps.append(end)
    # print('max end:', max(max(ends) for ends in end_list))
    # print('time_steps: ', time_steps)
    # print('min_interval: ', min_interval)
    # print('intervals//min: ', [sum(intervals)//min_interval for intervals in interval_list])
    # print('sum of intervals in each bar: ', [sum([round(intv/min_interval) for intv in intervals]) for intervals in interval_list])
    # print('remainder of division mod min_interval: ', [[round(intv%min_interval/min_interval) for intv in intervals] for intervals in interval_list])
    
    
    # create midi_array as an array of notes with a constant beat
    midi_array = []
    
    for i in range(len(notes_list)):
        k = 1
        voice = []
        # for j in range(len(notes_list[i])):
        #     length = round(interval_list[i][j] / min_interval)
        #     for l in range(length):
        #         k += 1
        #         voice.append(notes_list[i][j])
        for j in range(len(notes_list[i])):
            print('start of note: ', start_list[i][j])
            print('end of note: ', end_list[i][j])
            print('end of time_step: ', time_steps[k])
            print('gap from end of note to end of timestep: ', (end_list[i][j] - time_steps[k]) / min_interval)
            # check that the note has started within this timestep
            while (end_list[i][j] - time_steps[k]) / min_interval < 0.1: # i.e. while the note hasn't ended
                print('start of time_step: ', time_steps[k-1])
                if (start_list[i][j] - time_steps[k-1]) / min_interval < 0.1: # i.e. the note started within range of the end of the previous timestep
                    voice.append(notes_list[i][j])
                    k += 1
                else: # i.e. note has not started yet
                    voice.append(None) # append a pause
                    k += 1
        # print(voice)
        # add voice to midi array
        midi_array.append(voice)
    
    # print(midi_array)
    
    # check that all voices have same length
    lengths = []
    for voice in midi_array:
        lengths.append(len(voice))
        
    print('length of each voice: ',lengths)
    
    if not all_equal(lengths):
        raise Exception('Error in array output, not all voices have same length')
    
    # print(midi_array, min_interval)
    
    return midi_array, min_interval

# def midi_to_array(midi_file):
#     '''
#     Function to convert midi file to array for processing.
    
#     Parameters
#     ----------
#     midi_file : str
#         File path of the midi file

#     Returns
#     -------
#     midi_array : list
#         Array of notes (int) standardised to a constant beat

#     '''
#     midi_data = pretty_midi.PrettyMIDI(midi_file)
#     interval_list = []
#     start_list = []
#     end_list = []
#     notes_list = []
#     instruments =[]
    
#     for instrument in midi_data.instruments:
#         instruments.append(instrument.program)
#         notes = []
#         intervals = []
#         starts = []
#         ends = []
#         for note in instrument.notes:
#             interval = round((note.end - note.start) * 1000) # calculate the interval for standardisation (int in ms)
#             intervals.append(interval)
#             starts.append(round(note.start * 1000))
#             ends.append(round(note.end * 1000))
#             notes.append(note.pitch)
        
#         start_list.append(starts)
#         end_list.append(ends)
            
#         # add the voice to the notes_list array
#         notes_list.append(notes)
        
#         # add the voice to the notes_list array
#         interval_list.append(intervals)
           
#     # find minimum interval
#     min_interval = min(min(interval_list[i]) for i in range(4))
    
#     # create a set of timesteps
#     time_steps = [min(min(starts) for starts in start_list)]
#     end = min(min(starts) for starts in start_list)
#     while end < max(max(ends) for ends in end_list):
#         end += min_interval
#         time_steps.append(end)
#     # print('max end:', max(max(ends) for ends in end_list))
#     # print('time_steps: ', time_steps)
#     # print('min_interval: ', min_interval)
#     # print('intervals//min: ', [sum(intervals)//min_interval for intervals in interval_list])
#     # print('sum of intervals in each bar: ', [sum([round(intv/min_interval) for intv in intervals]) for intervals in interval_list])
#     # print('remainder of division mod min_interval: ', [[round(intv%min_interval/min_interval) for intv in intervals] for intervals in interval_list])
    
    
#     # create midi_array as an array of notes with a constant beat
#     midi_array = []
    
#     for i in range(len(notes_list)):
#         k = 1
#         voice = []
#         # for j in range(len(notes_list[i])):
#         #     length = round(interval_list[i][j] / min_interval)
#         #     for l in range(length):
#         #         k += 1
#         #         voice.append(notes_list[i][j])
#         for j in range(len(notes_list[i])):
#             print('start of note: ', start_list[i][j])
#             print('end of note: ', end_list[i][j])
#             print('end of time_step: ', time_steps[k])
#             print('gap from end of note to end of timestep: ', (end_list[i][j] - time_steps[k]) / min_interval)
#             # check that the note has started within this timestep
#             while (end_list[i][j] - time_steps[k]) / min_interval < 0.1: # i.e. while the note hasn't ended
#                 print('start of time_step: ', time_steps[k-1])
#                 if (start_list[i][j] - time_steps[k-1]) / min_interval < 0.1: # i.e. the note started within range of the end of the previous timestep
#                     voice.append(notes_list[i][j])
#                     k += 1
#                 else: # i.e. note has not started yet
#                     voice.append(None) # append a pause
#                     k += 1
#         # print(voice)
#         # add voice to midi array
#         midi_array.append(voice)
    
#     # print(midi_array)
    
#     # check that all voices have same length
#     lengths = []
#     for voice in midi_array:
#         lengths.append(len(voice))
        
#     print('length of each voice: ',lengths)
    
#     if not all_equal(lengths):
#         raise Exception('Error in array output, not all voices have same length')
    
#     # print(midi_array, min_interval)
    
#     return midi_array, min_interval

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
            # check if pause, i.e. note is empty
            if midi_array == None:
                # skip a beat, extend the time
                time += beat/1000
            else:
                # Create a Note instance, starting at 0 and ending at 1 beat (in s)
                note = pretty_midi.Note(
                    velocity=127, pitch = note_number, start = time, end = time + beat/1000)
            # Add it to our cello instrument
            instrument.notes.append(note)
            # extend the time
            time += beat/1000
        # Add the instrument to the PrettyMIDI object
        midi_output.instruments.append(instrument)
        
    # Write out the MIDI data
    midi_output.write(dest_file_path)
    
    return