from docplex.cp.model import CpoModel
import src.music_functions

class CPModel:
    def __init__(self, model_name, musical_input, chord_vocab):
        self.name = model_name #string
        self.musical_input = musical_input #An instance of the class MusicalWorkInput
        self.chord_vocab = chord_vocab #A list of objects, each of the class Chord
        self.N = self.musical_input.melody_len
        self.K = self.musical_input.key
        
        #Initialising Model
        self.m = CpoModel(name=self.name)
        
    def define_decision_variables(self):
        arr = [(i,j) for i in range(4) for j in range(self.N)]
        #i = 0 refers to soprano, 1 refers to alto, 2 refers to tenor, 3 refers to bass
        self.x = self.m.integer_var_dict(arr, name = "Notes")
        self.c = self.m.integer_var_list(self.N, min = 0, max = len(self.chord_vocab) - 1, name = "Chords")
        
    def hard_constraint_musical_input(self):
        for j in range(self.N):
            self.m.add(self.x[0,j] == self.musical_input.melody[j])
    
    def hard_constraint_voice_ranges(self, lb = [19, 12, 5], ub = [38, 28, 26]):
        #voice_ranges = {1: (19, 38), 2: (12, 28), 3: (5, 26)}
        for i in range(1,4):
            for j in range(self.N):
                self.m.add(self.x[i,j] >= lb[i-1])
                self.m.add(self.x[i,j] <= ub[i-1])
    
    def hard_constraint_chord_grades(self, lb = 5, ub = 60): #All notes must belong to the same chord
        chord_vocab_ext = []
        for chord in self.chord_vocab:
            chord_vocab_ext.append(src.music_functions.extend_range(src.music_functions.transpose(chord.note_intervals, self.K)))
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
            n1 = []
            for chord in self.chord_vocab:
                if chord.name == "i":
                    n = chord.index
                    n1.append = chord.index
                elif chord.name == "I":
                    n1.append = chord.index
            self.m.add(self.c[0] == n)
            self.m.add(self.c[self.N-1].set_domain(n1))
    
    def hard_constraint_adjacent_bar_chords(self):
        for j in range(1, self.N-1):
            if j % self.musical_input.meter == self.musical_input.first_on_beat:
                self.m.add(self.c[j-1] != self.c[j])
    
    def hard_constraint_voice_crossing(self):
        for i in range(3):
            for j in range(self.N):
                self.m.add(self.x[i,j] >= self.x[i+1,j])
    
    def add_hard_constraints(self, *args):
        for arg in args:
            arg
    
    def solve(self, log = True):
        sol = self.m.solve(log_output = log)
        print(sol.get_objective_values())       
        print(sol.print_solution())
        return sol