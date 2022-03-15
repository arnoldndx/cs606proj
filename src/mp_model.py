from docplex.mp.model import Model
from docplex.mp.progress import ProgressDataRecorder
import src.music_functions

class MPModel:
    def __init__(self, model_name, musical_input, chord_vocab, hard_constraints, soft_constraints, file_progression_cost):
        self.name = model_name #string
        self.musical_input = musical_input #An instance of the class MusicalWorkInput
        self.chord_vocab = chord_vocab #A list of objects, each of the class Chord
        self.N = self.musical_input.melody_len
        self.K = self.musical_input.key
        self.hard_constraints = hard_constraints #A dictionary with constraint names as key and boolean value on whether to include that constraint in the model or not
        self.soft_constraints = soft_constraints
        self.costs = {k: 0 for k in soft_constraints.keys()}
        
        # cost parameters, weights
        self.chord_progression_costs=func_get_progression_costs(file_progression_cost) # a dictionary
        
        #Initialising Model
        self.m = Model(name=self.name)
        
        #Decision Variables
        self.define_decision_variables()
        
        #Adding Constraints
        hard_constraints = {'musical input': self.hard_constraint_musical_input,
                            'voice range': self.hard_constraint_voice_range,
                            'chord membership': self.hard_constraint_chord_membership,
                            'first last chords': self.hard_constraint_first_last_chords,
                            'chord bass repetition': self.hard_constraint_chord_bass_repetition,
                            'adjacent bar chords': self.hard_constraint_adjacent_bar_chords,
                            'voice crossing': self.hard_constraint_voice_crossing,
                            'parallel movement': self.hard_constraint_parallel_movement,
                            'chord spacing': self.hard_constraint_chord_spacing}

        soft_constraints = {'chord progression': self.soft_constraint_chord_progression,
                            'chord bass repetition': self.soft_constraint_chord_bass_repetition,
                            'leap resolution': self.soft_constraint_leap_resolution,
                            'melodic movement': self.soft_constraint_melodic_movement,
                            'note repetition': self.soft_constraint_note_repetition,
                            'parallel movement': self.soft_constraint_parallel_movement,
                            'voice overlap': self.soft_constraint_voice_overlap,
                            'adjacent bar chords': self.soft_constraint_adjacent_bar_chords,
                            'chord spacing': self.soft_constraint_chord_spacing,
                            'distinct notes': self.soft_constraint_distinct_notes,
                            'voice crossing': self.soft_constraint_voice_crossing,
                            'voice range': self.soft_constraint_voice_range}
        for k, v in self.hard_constraints.items():
            if v:
                hard_constraints[k]()
        for k, v in self.soft_constraints.items():
            if v:
                self.costs[k] = soft_constraints[k]()
        
        #Objective Function
        self.m.minimize(m.sum(self.costs))
        
    def define_decision_variables(self):
        arr = [(i,j) for i in range(4) for j in range(self.N)]
        #i = 0 refers to soprano, 1 refers to alto, 2 refers to tenor, 3 refers to bass
        self.x = self.m.integer_var_dict(arr, name = "Notes")
        self.c = self.m.integer_var_list(self.N, min = 0, max = len(self.chord_vocab) - 1, name = "Chords")

    def hard_constraint_musical_input(self):
        for j in range(self.N):
            self.m.add_constraint(self.x[0,j] == self.musical_input.melody[j])
    
    def hard_constraint_voice_range(self, lb = [19, 12, 5], ub = [38, 28, 26]):
        #voice_ranges = {1: (19, 38), 2: (12, 28), 3: (5, 26)}
        for i in range(1,4):
            for j in range(self.N):
                self.m.add_constraint(self.x[i,j] >= lb[i-1])
                self.m.add_constraint(self.x[i,j] <= ub[i-1])
    
    def hard_constraint_chord_membership(self): #All notes must belong to the same chord
        chord_vocab_ext = []
        for chord in self.chord_vocab:
            chord_vocab_ext.append(src.music_functions.extend_range(src.music_functions.transpose(chord.note_intervals, self.K)))
            length=len(chord_vocab_ext[0])
        for j in range(self.N):
            for i in range(4):
                #for note in range(lb, ub):
                for chord, chord_ext in zip(self.chord_vocab, chord_vocab_ext):
                    self.m.add_constraint((self.c[j] == chord.index)<= self.m.sum((self.x[i,j]-24)==chord_ext[p] for p in range(length)] )
                    #self.m.add(self.m.if_then(self.m.logical_and(self.c[j] == chord.index, self.x[i,j] == note), note in chord_ext))
    
    def hard_constraint_first_last_chords(self): # what is the logic in CP model?
        Tonality= 1 if self.musical_input.tonality =="major" else 0
        self.m.add_constraint(Tonality ==self.c[0])
        self.m.add_constraint(self.c[self.N-1] <=1 )
        self.m.add_constraint(Tonality <=self.c[self.N-1])
        
    def hard_constraint_chord_bass_repetition(self):
       for j in range(self.N-1):
            self.m.add_constraint( (self.c[j] == self.c[j+1]) <= (self.x[3,j] != self.x[3,j+1]))
    
    def hard_constraint_adjacent_bar_chords(self):
        for j in range(1,self.N):
            if j % self.musical_input.meter == self.musical_input.first_on_beat:
                self.m.add_constraint(self.c[j] != self.c[j-1])
    
    def hard_constraint_voice_crossing(self):
        for i in range(3):
            for j in range(self.N):
                self.m.add_constraint(self.x[i,j] >= self.x[i+1,j])
    
    def hard_constraint_parallel_movement(self, disallowed_intervals = [7, 12]):
        for j in range(self.N-1):
            for i in range(3):
                for k in range(i+4):
                    for interval in disallowed_intervals:
                        self.m.add_constraint( (x[i,j]-x[k,j] ==interval)<= ( x[i,j+1]-x[k,j+1] !=interval) )
                        self.m.add_constraint( (x[i,j]-x[k,j] ==interval+12)<= ( x[i,j+1]-x[k,j+1] !=interval+12) )
                        self.m.add_constraint( (x[i,j]-x[k,j] ==interval+24)<= ( x[i,j+1]-x[k,j+1] !=interval+24) )

    def hard_constraint_chord_spacing(self, max_spacing = [12, 12, 16]):
        for j in range(self.N):
            for i in range(3):
                self.m.add_constraint(self.x[i,j] - self.x[i+1,j] <= max_spacing[i])
    
    def soft_constraint_chord_progression(self):
        cost0= self.m.continuous_var_list(self.N, 0,100, "Progression cost")
        length=len(self.chord_vocab)
        
        for j in range(self.N-1):
            for c1 in range(length):
                for c2 in range(length):
                    self.m.add_constraint(cost0[j]>=self.m.logical_and(c1==self.c[j],c2==self.c[j+1])*self.chord_progression_costs[(c1,c2)]) 
            
    def soft_constraint_leap_resolution(self):
        pass

    def soft_constraint_melodic_movement(self):
        pass
    
    def soft_constraint_note_repetition(self, weight):
        cost4= self.m.integer_var_list(self.N, 0,1, "Repetition cost")
        for j in range(N-2):
            self.m.add_constraint(cost4[j]>= weight *
                                  self.m.sum(self.m.logical_and(self.x[i,j]==self.x[i,j+1],self.x[i,j+1]==self.x[i,j+2]) for i in range(1,4))
                                  )

    def soft_constraint_parallel_movement(self):
        pass
    
    def soft_constraint_voice_overlap(self):
        pass
    
    def soft_constraint_chord_spacing(self):
        pass
    
    def soft_constraint_distinct_notes(self):
        pass
    
    def soft_constraint_voice_crossing(self):
        cost6= self.m.continuous_var_list(self.N, 0,100, "voice crossing cost")
        for i in range(3):
            for j in range(self.N-1):
                self.m.add_constraint(cost6[j]>= (self.x[i,j+1] <= self.x[i+1,j]-1))
    
    def soft_constraint_voice_range(self, slb = [24,17, 10], sub = [33,23 ,21]):
        cost6_1= self.m.continuous_var_list(self.N, 0,15, "voice range cost")
        cost6_2= self.m.continuous_var_list(self.N, 0,15, "voice range cost")
        
        for j in range(self.N):
            cost6_1[j]>=self.m.sum( self.x[i,j] -sub[i-1]  for i in range(1,4))
            cost6_2[j]>=self.m.sum( slb[i-1]-self.x[i,j]  for i in range(1,4))
            
                        
    def solve(self, log = True):
        recorder = ProgressDataRecorder()
        self.m.add_progress_listener(recorder)
        sol = self.m.solve(log_output = log)
        #print(sol.get_objective_values())       
        print(sol)
        return sol
        
        
    #
    # best_bound = []
# det_time = []
# current_objective = []
# for data in recorder.recorded:
    # best_bound.append(data.best_bound)
    # det_time.append(data.det_time/100)
    
    
    # current_objective.append(data.current_objective)
    
    # plt.plot(det_time, current_objective, label = "best integer", marker='s', linestyle = '--')
# plt.plot(det_time, best_bound, label = "best node")
# plt.legend()
# plt.show()