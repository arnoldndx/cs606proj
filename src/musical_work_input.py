import src.music_functions
import src.evaluate_v0
import src.evaluate
import sys

sys.path.append('../')


from src.ALNS.alns import ALNS, State
import copy
class MusicalWorkInput:
    def __init__(self, title, meter, key, tonality, first_on_beat, melody, reference_note = 24):
        self.title = title #string
        self.meter = meter #integer
        self.key = key #integer, where 0 refers to C, 1 refers to C#/Db, ..., 11 refers to B
        self.tonality = tonality #'major' or 'minor'
        self.first_on_beat = first_on_beat #integer not exceeding meter
        self.melody = melody #list of integers
        self.reference_note = reference_note #refers to which integer corresponds to middle C, i.e. C4
        self.melody_len = len(self.melody)
    
    def transpose_work(self, n_semitones, ascending):
        self.melody = src.music_functions.transpose(self.melody, n_semitones, mod = False, ascending = ascending)
        self.key = src.music_functions.transpose(self.key, n_semitones, mod = True, ascending = ascneding)

    def update_reference_note(self, new_reference_note):
        diff = new_reference_note - self.reference_note
        self.transpose_work(diff, ascending = True)
        self.reference_note = new_reference_note
        
        
class Harmony(State):
    def __init__(self, MusicInput, HarmonyInput):
    #MusicInput: a MusicalWorkInput object
    #HarmonyInput:2-dim list (5*N) of all 4 parts and chord index, can be incomplete (missing value is -100)
        self.MusicInput = MusicInput
        self.HarmonyInput=HarmonyInput
        self.HarmonyOutput=HarmonyInput  # to be modified
        self.notes=HarmonyInput[:-1]
        self.chords=HarmonyInput[-1]
        self.N=len(HarmonyInput[0])
        assert len(HarmonyInput)==5
    def copy(self):
        return copy.deepcopy(self)
    
    def iscomplete(self):
        return sum (self.HarmonyInput[i][j]<=-99 for i in range (5) for j in range(self.N)) < 0.01
    
    def get_cost_list(self):
        return src.evaluate.evaluate_cost(self.notes, self.chords, self.MusicInput.key, self.MusicInput.tonality, 
                                          self.MusicInput.meter, self.MusicInput.first_on_beat,
                                          mode="L")
    def objective(self): #get_cost_sum()    
        cost_list= src.evaluate.evaluate_cost(self.notes, self.chords, self.MusicInput.key, self.MusicInput.tonality, 
                                              self.MusicInput.meter, self.MusicInput.first_on_beat,
                                              mode="D")
        return sum( v for k,v in cost_list.items() if k[:4]=="soft")

#def evaluate_cost(list_x, list_c , tonality, meter=4, first_on_beat=0, mode="L") 