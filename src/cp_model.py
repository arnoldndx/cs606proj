from docplex.cp.model import CpoModel
from src.music_functions import *
import numpy as np
from src.midi_processing import *

class CPModel:
    def __init__(self, model_name, musical_input, chord_vocab, chord_progression_penalties, hard_constraints, soft_constraints_weights):
        self.name = model_name #string
        self.musical_input = musical_input #An instance of the class MusicalWorkInput
        self.chord_vocab = chord_vocab #A list of objects, each of the class Chord
        self.chord_progression_penalties = chord_progression_penalties #A dictionary of (chord1 name, chord2 name) as the key and penalty as value
        self.N = self.musical_input.melody_len
        self.K = self.musical_input.key
        self.hard_constraints = hard_constraints #A dictionary with constraint names as key and boolean value on whether to include that constraint in the model or not
        self.soft_constraints_weights = soft_constraints_weights
        self.hard_constraint_encoding, self.soft_constraint_encoding = encode_constraints(hard_constraints, soft_constraints_weights)
        self.costs = {k: 0 for k in soft_constraints_weights.keys()}
        
        #Initialising Model
        self.m = CpoModel(name=self.name)
        
        #Decision Variables
        self.define_decision_variables()
        
        #Adding Constraints
        hard_constraints = {'musical input': self.hard_constraint_musical_input,
                            'voice range': self.hard_constraint_voice_range,
                            'chord membership': self.hard_constraint_chord_membership,
                            'first last chords': self.hard_constraint_first_last_chords,
                            'chord repetition': self.hard_constraint_chord_repetition,
                            'chord bass repetition': self.hard_constraint_chord_bass_repetition,
                            'adjacent bar chords': self.hard_constraint_adjacent_bar_chords,
                            'voice crossing': self.hard_constraint_voice_crossing,
                            'parallel movement': self.hard_constraint_parallel_movement,
                            'chord spacing': self.hard_constraint_chord_spacing,
                            'incomplete chord': self.hard_constraint_incomplete_chord,
                            'chord distribution': self.hard_constraint_chord_distribution}

        soft_constraints = {'chord progression': self.soft_constraint_chord_progression,
                            'chord repetition': self.soft_constraint_chord_repetition,
                            'chord bass repetition': self.soft_constraint_chord_bass_repetition,
                            'leap resolution': self.soft_constraint_leap_resolution,
                            'melodic movement': self.soft_constraint_melodic_movement,
                            'note repetition': self.soft_constraint_note_repetition,
                            'parallel movement': self.soft_constraint_parallel_movement,
                            'voice overlap': self.soft_constraint_voice_overlap,
                            'adjacent bar chords': self.soft_constraint_adjacent_bar_chords,
                            'chord spacing': self.soft_constraint_chord_spacing,
                            'distinct notes': self.soft_constraint_distinct_notes,
                            'incomplete chord': self.soft_constraint_incomplete_chord,
                            'voice crossing': self.soft_constraint_voice_crossing,
                            'voice range': self.soft_constraint_voice_range,
                            'second inversion': self.soft_constraint_second_inversion,
                            'first inversion': self.soft_constraint_first_inversion,
                            'chord distribution': self.soft_constraint_chord_distribution}
        for k, v in self.hard_constraints.items():
            if v:
                hard_constraints[k]()
        for k, v in self.soft_constraints_weights.items():
            if v > 0:
                soft_constraints[k]()
        
        #Objective Function
        self.m.minimize(self.m.sum(self.costs[k][i,j] for i in range(4) for j in range(self.N) for k in self.soft_constraints_weights))
        
    def define_decision_variables(self):
        arr = [(i,j) for i in range(4) for j in range(self.N)]
        #i = 0 refers to soprano, 1 refers to alto, 2 refers to tenor, 3 refers to bass
        self.x = self.m.integer_var_dict(arr, name = "Notes")
        self.c = self.m.integer_var_list(self.N, min = 0, max = len(self.chord_vocab) - 1, name = "Chords")
        for k, v in self.soft_constraints_weights.items():
            self.costs[k] = self.m.integer_var_dict(arr, min = 0, name = k)

    def hard_constraint_musical_input(self):
        for j in range(self.N):
            self.m.add(self.x[0,j] == self.musical_input.melody[j])
    
    def hard_constraint_voice_range(self, lb = [19, 12, 5], ub = [38, 28, 26]):
        #voice_ranges = {1: (19, 38), 2: (12, 28), 3: (5, 26)}
        for i in range(1,4):
            for j in range(self.N):
                self.m.add(self.x[i,j] >= lb[i-1])
                self.m.add(self.x[i,j] <= ub[i-1])
    
    def hard_constraint_chord_membership(self, lb = 5, ub = 60): #All notes must belong to the same chord
        chord_vocab_ext = []
        for chord in self.chord_vocab:
            chord_vocab_ext.append(extend_range(transpose(chord.note_intervals, self.K)))
        for j in range(self.N):
            for i in range(4):
                for note in range(lb, ub):
                    for chord, chord_ext in zip(self.chord_vocab, chord_vocab_ext):
                        self.m.add(self.m.if_then(self.m.logical_and(self.c[j] == chord.index, self.x[i,j] == note), note in chord_ext))
    
    def hard_constraint_first_last_chords(self):
        if self.musical_input.tonality == "major":
            for chord in self.chord_vocab:
                if chord.name == "I":
                    n = chord.index
                    break
            self.m.add(self.c[0] == n)
            self.m.add(self.c[self.N-1] == n)
        elif self.musical_input.tonality == "minor":
            for chord in self.chord_vocab:
                if chord.name == "i":
                    self.m.add(self.c[0] == chord.index)
                elif chord.name != "I":
                    self.m.add(self.c[self.N-1] != chord.index)
        
        #First and last bass notes must be the tonic note
        self.m.add(self.x[3,0] % 12 == self.K)
        self.m.add(self.x[3,self.N-1] % 12 == self.K)
    
    def hard_constraint_chord_repetition(self):
        for j in range(self.N-1):
            self.m.add(self.c[j+1] != self.c[j])
    
    def hard_constraint_chord_bass_repetition(self):
        for j in range(self.N-1):
            self.m.add(self.m.if_then(self.c[j] == self.c[j+1], self.x[3,j] != self.x[3,j+1]))
    
    def hard_constraint_adjacent_bar_chords(self):
        for j in range(1,self.N):
            if j % self.musical_input.meter == self.musical_input.first_on_beat:
                self.m.add(self.c[j] != self.c[j-1])
    
    def hard_constraint_voice_crossing(self):
        for i in range(3):
            for j in range(self.N):
                self.m.add(self.x[i,j] >= self.x[i+1,j])
    
    def hard_constraint_parallel_movement(self, disallowed_intervals = [0, 7]):
        for j in range(self.N-1):
            for i1 in range(4):
                for i2 in range(4):
                    if i2 != i1:
                        for interval in disallowed_intervals:
                            self.m.add(self.m.if_then(self.m.logical_and(
                                self.m.logical_and(self.x[i1,j] >= self.x[i2,j], self.x[i1,j+1] >= self.x[i2,j+1]),
                                (self.x[i1,j] - self.x[i2,j])%12 == interval),
                                                      (self.x[i1,j+1] - self.x[i2,j+1])%12 != interval))

    def hard_constraint_chord_spacing(self, max_spacing = [12, 12, 16]):
        for j in range(self.N):
            for i in range(3):
                self.m.add(self.m.abs(self.x[i,j] - self.x[i+1,j]) <= max_spacing[i])

    def hard_constraint_incomplete_chord(self): #The 4 voices must fully cover the 3 notes in a chord
        for j in range(self.N):
            for chord in self.chord_vocab:
                for note in chord.note_intervals:
                    note = (note + self.K) % 12
                    self.m.add(self.m.if_then(self.c[j] == chord.index,
                                              self.m.logical_or(
                                                  self.m.logical_or(self.x[0,j] % 12 == note, self.x[1,j] % 12 == note),
                                                  self.m.logical_or(self.x[2,j] % 12 == note, self.x[3,j] % 12 == note))))

    def hard_constraint_chord_distribution(self): #Activating this may result in no feasible solution.
        #Distance between adjacent lower voices must not be less than distance between adjacent higher voices.
        for j in range(self.N):
            for i in range(2):
                self.m.add(self.x[i,j] - self.x[i+1,j] <= self.x[i+1,j] - self.x[i+2,j])
    
    def soft_constraint_chord_progression(self):
        d = self.chord_progression_penalties
        w = self.soft_constraints_weights['chord progression']
        for j in range(self.N-1):
            for chord1 in self.chord_vocab:
                for chord2 in self.chord_vocab:
                    self.m.add(self.m.if_then(self.m.logical_and(self.c[j] == chord1.index, self.c[j+1] == chord2.index),
                                              self.costs['chord progression'][0,j] >= d[chord1.name, chord2.name] * w))

    def soft_constraint_chord_repetition(self):
        for j in range(self.N-1):
            self.m.add(self.m.if_then(self.c[j] == self.c[j+1],
                                      self.costs['chord repetition'][0,j] >= self.soft_constraints_weights['chord repetition']))

    def soft_constraint_chord_bass_repetition(self):
        for j in range(self.N-1):
            self.m.add(self.m.if_then(self.m.logical_and(self.c[j] == self.c[j+1], self.x[3,j] == self.x[3,j+1]),
                                      self.costs['chord bass repetition'][3,j] >= self.soft_constraints_weights['chord bass repetition']))
        
    
    def soft_constraint_leap_resolution(self):
        pass

    def soft_constraint_melodic_movement(self, leap_interval = {1: 12, 2: 12, 3: 16}):
        for j in range(self.N-1):
            for i in range(1,4):
                self.m.add(self.costs['melodic movement'][i,j] >= self.m.abs(self.x[i,j+1] - self.x[i,j]) / 12 * self.soft_constraints_weights['melodic movement'])
    
    def soft_constraint_note_repetition(self):
        for j in range(self.N-2):
            for i in range(4):
                self.m.add(self.m.if_then(self.m.logical_and(self.x[i,j] == self.x[i,j+1], self.x[i,j+1] == self.x[i,j+2]),
                                          self.costs['note repetition'][i,j] >= self.soft_constraints_weights['note repetition']))
    
    def soft_constraint_parallel_movement(self, discouraged_intervals = [0, 7]):
        for j in range(self.N-1):
            for i1 in range(4):
                for i2 in range(4):
                    for interval in discouraged_intervals:
                        self.m.add(self.m.if_then(self.m.logical_and(
                            self.m.logical_and(self.x[i1,j] >= self.x[i2,j], self.x[i1,j+1] >= self.x[i2,j+1]),
                            self.m.logical_and((self.x[i1,j] - self.x[i2,j])%12 == interval,
                            (self.x[i1,j+1] - self.x[i2,j+1])%12 == interval)),
                                   self.costs['parallel movement'][i1,j] >= self.soft_constraints_weights['parallel movement']))
    
    def soft_constraint_voice_overlap(self):
        for j in range(self.N-1):
            for i1 in range(4):
                for i2 in range(4):
                    if i2 - i1 == 1:
                        self.m.add(self.m.if_then(self.m.logical_and(self.x[i1,j] > self.x[i2,j], self.x[i2,j+1] > self.x[i1,j]),
                                                  self.costs['voice overlap'][i1,j] >= self.soft_constraints_weights['voice overlap']))
    
    def soft_constraint_adjacent_bar_chords(self):
        for j in range(1,self.N):
            if j % self.musical_input.meter == self.musical_input.first_on_beat:
                self.m.add(self.m.if_then(self.c[j] == self.c[j-1],
                           self.costs['adjacent bar chords'][0,j] >= self.soft_constraints_weights['adjacent bar chords']))
    
    def soft_constraint_chord_spacing(self, max_spacing = [12, 12, 16]):
        for j in range(self.N):
            for i in range(3):
                self.m.add(self.m.if_then(self.x[i,j] - self.x[i+1,j] > max_spacing[i],
                           self.costs['chord spacing'][i+1,j] >= self.soft_constraints_weights['chord spacing']))
    
    def soft_constraint_distinct_notes(self):
        for j in range(self.N):
            for i1 in range(4):
                for i2 in range(4):
                    for i3 in range(4):
                        if i1 < i2 and i2 < i3:
                            #To penalise chords with 3 repeated notes
                            self.m.add(self.m.if_then(self.m.logical_and(
                                self.x[i1,j] % 12 == self.x[i2,j] % 12,
                                self.x[i2,j] % 12 == self.x[i3,j] % 12),
                                                      self.costs['distinct notes'][0,j] >= self.soft_constraints_weights['distinct notes']))
                    if i1 < i2:
                        for chord in self.chord_vocab:
                            note = chord.note_intervals[1]
                            note = (note + self.K) % 12
                            #To penalise chords with a doubled 3rd note (i.e. 2nd value in the note intervals)
                            self.m.add(self.m.if_then(self.m.logical_and(
                                self.c[j] == chord.index,
                                self.m.logical_and(self.x[i1,j] % 12 == self.x[i2,j] % 12,
                                                   self.x[i1,j] % 12 == note)),
                                                      self.costs['distinct notes'][1,j] >= self.soft_constraints_weights['distinct notes']))

    def soft_constraint_incomplete_chord(self):
        for j in range(self.N):
            for chord in self.chord_vocab:
                #To penalise chords with missing notes
                for k, note in enumerate(chord.note_intervals):
                    note = (note + self.K) % 12
                    self.m.add(self.m.if_then(
                        self.m.logical_and(self.c[j] == chord.index,
                                           self.m.logical_and(
                                               self.m.logical_and(self.x[0,j] % 12 != note, self.x[1,j] % 12 != note),
                                               self.m.logical_and(self.x[2,j] % 12 != note, self.x[3,j] % 12 != note))),
                        self.costs['incomplete chord'][k,j] >= self.soft_constraints_weights['incomplete chord']))
    
    def soft_constraint_voice_crossing(self):
        for i in range(3):
            for j in range(self.N):
                self.m.add(self.m.if_then(self.x[i,j] < self.x[i+1,j],
                                        self.costs['voice crossing'][1,j] >= self.soft_constraints_weights['voice crossing']))
    
    def soft_constraint_voice_range(self, lb = [19, 12, 5], ub = [38, 28, 26], threshold = 2):
        for i in range(1,4):
            for j in range(self.N):
                self.m.add(self.m.if_then(self.m.logical_or(self.x[i,j] < lb[i-1] + threshold, self.x[i,j] > ub[i-1] - threshold),
                                          self.costs['voice range'][i,j] >= self.soft_constraints_weights['voice range']))

    def soft_constraint_second_inversion(self):
        for chord in self.chord_vocab:
            note = chord.note_intervals[-1]
            note = (note + self.K) % 12
            for j in range(1,self.N-1): #Excluding first and last chord
                #Penalising 2nd inversion chords if it is not used as a passing chord
                '''
                self.m.add(self.m.if_then(self.m.logical_and(self.c[j] == chord.index, self.x[3,j] % 12 == note), #This identifies whether chod is a 2nd inversion
                                          self.m.if_then(self.m.logical_not(self.m.logical_or(
                                              self.m.logical_and(self.m.logical_and(self.x[3,j-1] - self.x[3,j] <= 2, self.x[3,j-1] - self.x[3,j] > 0),
                                                                 self.m.logical_and(self.x[3,j] - self.x[3,j+1] <= 2, self.x[3,j] - self.x[3,j+1] > 0)), #This identifies stepwise downward motion
                                              self.m.logical_and(self.m.logical_and(self.x[3,j] - self.x[3,j-1] <= 2, self.x[3,j] - self.x[3,j-1] > 0),
                                                                 self.m.logical_and(self.x[3,j+1] - self.x[3,j] <= 2, self.x[3,j+1] - self.x[3,j] > 0)))), #This identifies stepwise upward motion
                                                         self.costs['second inversion'][3,j] >= self.soft_constraints_weights['second inversion'])))'''
                #Just penalising all 2nd inversion chords
                self.m.add(self.m.if_then(self.m.logical_and(self.c[j] == chord.index, self.x[3,j] % 12 == note),
                                          self.costs['second inversion'][3,j] >= self.soft_constraints_weights['second inversion']))
            #Special cases - First and last chords #Not included because of hard_constraint_first_last_chord. Consider uncommenting the following if first last chord constraint is excluded
            '''for j in [0, self.N-1]:
                self.m.add(self.m.if_then(self.m.logical_and(self.c[j] == chord.index, self.x[3,j] % 12 == note),
                                          self.costs['second inversion'][3,j] >= self.soft_constraints_weights['second inversion']))'''

    def soft_constraint_first_inversion(self):
        for chord in self.chord_vocab:
            note = chord.note_intervals[1]
            note = (note + self.K) % 12
            for j in range(1,self.N-1): #Excluding first and last chord
                #Penalising all 1st inversion chords
                self.m.add(self.m.if_then(self.m.logical_and(self.c[j] == chord.index, self.x[3,j] % 12 == note),
                                          self.costs['first inversion'][3,j] >= self.soft_constraints_weights['first inversion']))
            #Special cases - First and last chords #Not included because of hard_constraint_first_last_chord. Consider uncommenting the following if first last chord constraint is excluded
            '''for j in [0, self.N-1]:
                self.m.add(self.m.if_then(self.m.logical_and(self.c[j] == chord.index, self.x[3,j] % 12 == note),
                                          self.costs['first inversion'][3,j] >= self.soft_constraints_weights['first inversion']))'''

    def soft_constraint_chord_distribution(self):
        #Penalise if the distance between adjacent higher voices is greater than distance between adjacent lower voices.
        for j in range(self.N):
            for i in range(2):
                self.m.add(self.m.if_then(self.x[i,j] - self.x[i+1,j] > self.x[i+1,j] - self.x[i+2,j],
                                          self.costs['chord distribution'][i,j] >= self.soft_constraints_weights['chord distribution']))
                

    def solve(self, **kwargs):
        sol = self.m.solve(**kwargs)
        print(sol.get_objective_values())
        print(sol.print_solution())
        self.sol = sol
        return sol
    
    def get_solution(self):
        chord_var_names = ['Chords_{}'.format(str(j)) for j in range(self.N)]
        note_var_names = ['Notes_{}'.format(str(j)) for j in range(self.N * 4)]
        penalty_names = {k: ['{}_{}'.format(k, str(j)) for j in range(self.N * 4)] for k in self.soft_constraints_weights}
        chord_sol = [self.sol.get_value(x) for x in chord_var_names]
        chord_sol = [self.chord_vocab[chord].name for chord in chord_sol]
        note_sol = [[self.sol.get_value(x) for x in note_var_names[i*self.N:(i+1)*self.N]] for i in range(4)]
        penalties = {k: [[self.sol.get_value(x) for x in penalty_names[k][i*self.N:(i+1)*self.N]] for i in range(4)] for k in self.soft_constraints_weights}
        self.sol_var = {'Chords': chord_sol, 'Notes': note_sol, 'Penalties': penalties}
        return self.sol_var
        
    def export_midi(self, instruments = [20]*4, beat = 500, filepath = '../outputs'):
        array_to_midi(self.sol_var['Notes'], instruments, beat,
                      dest_file_path = '{}/cp_{}_{}_{}_{}.mid'.format(
                          filepath, self.name, self.musical_input.title, self.hard_constraint_encoding, self.soft_constraint_encoding),
                     held_notes = True, offset = (self.musical_input.meter - self.musical_input.first_on_beat + 1) % 4)