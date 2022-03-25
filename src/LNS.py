from docplex.mp.model import Model
from docplex.mp.progress import ProgressDataRecorder
import src.music_functions
import matplotlib.pyplot as plt


class LNS_Model:
    def __init__(self, model_name, musical_input, chord_vocab
                 # , hard_constraints
                 , soft_constraint_w_weights
                 , file_progression_cost):
        self.name = model_name  # string
        self.musical_input = musical_input  # An instance of the class MusicalWorkInput
        self.chord_vocab = chord_vocab  # A list of objects, each of the class Chord
        self.N = self.musical_input.melody_len
        self.K = self.musical_input.key
        # self.hard_constraints = hard_constraints #A dictionary with constraint names as key and boolean value on whether to include that constraint in the model or not
        self.soft_constraints_w_weights = soft_constraint_w_weights
        self.x = {}
        self.c = []

        # cost parameters, weights
        self.chord_progression_costs = src.music_functions.func_get_progression_costs(
            file_progression_cost)  # a dictionary

        self.define_decision_variables()

        self.construct_Initial_solution()

    def construct_Initial_solution(self):
        self.c[0] = 0
        self.c[self.N-1] = 0
        for n in range(0,self.N):
            this_note = self.musical_input.melody[n]
            self.x[0,n] = this_note
            self.x[3,n] = this_note - 24
            self.x[2, n] = this_note - 20
            self.x[1, n] = this_note - 17
            self.c = this_note%12

    def define_decision_variables(self):
        arr = [(i,j) for i in range(4) for j in range(self.N)]
        #i = 0 refers to soprano, 1 refers to alto, 2 refers to tenor, 3 refers to bass
        self.x = dict.fromkeys(arr) #x[0,n] is the melody. x[1,n], x[2,n] and x[3,n] are the chord notes.
        self.c = []

