from docplex.mp.model import Model
from docplex.mp.progress import ProgressDataRecorder
import src.music_functions 
import matplotlib.pyplot as plt

class MPModel:
    def __init__(self, model_name, musical_input, chord_vocab
                 #, hard_constraints
                 , soft_constraint_w_weights
                 , file_progression_cost):
        self.name = model_name #string
        self.musical_input = musical_input #An instance of the class MusicalWorkInput
        self.chord_vocab = chord_vocab #A list of objects, each of the class Chord
        self.N = self.musical_input.melody_len
        self.K = self.musical_input.key
        #self.hard_constraints = hard_constraints #A dictionary with constraint names as key and boolean value on whether to include that constraint in the model or not
        self.soft_constraints_w_weights = soft_constraint_w_weights
        
        
        # cost parameters, weights
        self.chord_progression_costs=src.music_functions.func_get_progression_costs(file_progression_cost) # a dictionary
        
        #Initialising Model
        self.m = Model(name=self.name)
        self.m.context.update_cplex_parameters({'randomseed': 606, 'mip.tolerances.mipgap': 0.002,'timelimit': 300})
        #self.m.context.update_cplex_parameters({'randomseed': 606, 'mip.tolerances.mipgap': 0.002,'timelimit': 120})
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
                             'chord spacing': self.hard_constraint_chord_spacing
                            }

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
        #self.costs = {k: 0 for k in soft_constraints.keys()}
        # for k, v in self.hard_constraints.items():
        #     if v:
        #         hard_constraints[k]()
        # for k, v in self.soft_constraints.items():
        #     if v:
        #         self.costs[k]
        for k in hard_constraints:
            
            hard_constraints[k]()
        counter=0
        self.costs=[]
        for k,v in self.soft_constraints_w_weights.items():
            if v>0 and k in soft_constraints:
            # if weight input >0 the constraint is turned on  
                self.costs.append(soft_constraints[k](weight=v))
                counter+=1

        #Objective Function
        self.m.minimize(self.m.sum(self.costs[p][j] for p in range(counter) for j in range(self.N))  )

    def define_decision_variables(self):
        arr = [(i,j) for i in range(4) for j in range(self.N)]
        #i = 0 refers to soprano, 1 refers to alto, 2 refers to tenor, 3 refers to bass
        self.x = self.m.integer_var_dict(arr, name = "Notes")
        self.c = self.m.integer_var_list(self.N, 0, len(self.chord_vocab) - 1, name = "Chords")

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
        offset=self.musical_input.reference_note
        chord_vocab_ext = []
        for chord in self.chord_vocab:
            chord_vocab_ext.append(src.music_functions.extend_range(src.music_functions.transpose(chord.note_intervals, self.K)))
            length=len(chord_vocab_ext[0])
            
        for j in range(self.N):
            for i in range(4):
                #for note in range(lb, ub):
                for chord, chord_ext in zip(self.chord_vocab, chord_vocab_ext):
                    self.m.add_constraint((self.c[j] == chord.index) <= self.m.sum((self.x[i,j]-offset==chord_ext[p]) for p in range(length) ))
                    #self.m.add(self.m.if_then(self.m.logical_and(self.c[j] == chord.index, self.x[i,j] == note), note in chord_ext))
    
    def hard_constraint_first_last_chords(self): # what is the logic in CP model?

        
        if self.musical_input.tonality == "major":
            for chord in self.chord_vocab:
                if chord.name == "I":
                    n = chord.index
                    break
            self.m.add_constraint(self.c[0] == n)
            self.m.add_constraint(self.c[self.N-1] == n)
        elif self.musical_input.tonality == "minor":
            n1 = []
            for chord in self.chord_vocab:
                if chord.name == "i":
                    n = chord.index
                    n1.append (chord.index)
                elif chord.name == "I":
                    n1.append (chord.index)
            self.m.add_constraint(self.c[0] == n)
            self.m.add_constraint(self.c[self.N-1]<=max(n1))
        
        
        
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
                for k in range(i+1, 4):
                    for interval in disallowed_intervals:
                        self.m.add_constraint( (self.x[i,j]-self.x[k,j] ==interval)<= ( self.x[i,j+1]-self.x[k,j+1] !=interval) )
                        self.m.add_constraint( (self.x[i,j]-self.x[k,j] ==interval+12)<= (self. x[i,j+1]-self.x[k,j+1] !=interval+12) )
                        self.m.add_constraint( (self.x[i,j]-self.x[k,j] ==interval+24)<= ( self.x[i,j+1]-self.x[k,j+1] !=interval+24) )

    def hard_constraint_chord_spacing(self, max_spacing = [12, 12, 16]):
        for j in range(self.N):
            for i in range(3):
                self.m.add_constraint(self.x[i,j] - self.x[i+1,j] <= max_spacing[i-1])
                
                
#***************************************************************************************************************    
    def soft_constraint_chord_progression(self, weight=1):
        cost0= self.m.continuous_var_list(self.N, 0,100, "Progression cost")
        length=len(self.chord_vocab)
                
        for j in range(self.N-1):
            for c1 in range(length):
                for c2 in range(length):
                    self.m.add_constraint(cost0[j]>=self.m.logical_and(c1==self.c[j],c2==self.c[j+1])*self.chord_progression_costs[(c1,c2)]) 
        return cost0 
    def soft_constraint_leap_resolution(self,max_leap=10, weight=1): # leaps more than an interval of a major 6th should resolve in the opposite direction by stepwise motion(
        cost1= self.m.continuous_var_list(self.N, 0,100, "Leap resolution cost")
        for j in range(self.N-1):
            self.m.add_constraint(cost1[j]>=self.m.sum(1-(self.x[i,j] - self.x[i,j+1] <= max_leap) for i in range(1,4) ) )
        return cost1                 

    def soft_constraint_melodic_movement(self):
        pass
    def soft_constraint_chord_bass_repetition(self):
        pass
    def soft_constraint_adjacent_bar_chords(self):
        pass
    
    
    def soft_constraint_note_repetition(self, weight=2):
        cost3= self.m.continuous_var_list(self.N, 0,100, "Repetition cost")
        for j in range(self.N-2):
            self.m.add_constraint(cost3[j]>= weight *
                                  self.m.sum(self.m.logical_and(self.x[i,j]==self.x[i,j+1],self.x[i,j+1]==self.x[i,j+2]) for i in range(1,4))
                                  )
        return cost3

    def soft_constraint_parallel_movement(self):# is a hard constraint
        pass
    
    def soft_constraint_voice_overlap(self,weight=1):
        cost4= self.m.continuous_var_list(self.N, 0,100, "voice crossing cost")
        for i in range(3):
            for j in range(self.N-1):
                self.m.add_constraint(cost4[j]>= (self.x[i,j+1] <= self.x[i+1,j]-1))
        return cost4
    def soft_constraint_chord_spacing(self, weight=1):
        pass
    
    def soft_constraint_distinct_notes(self,weight=1):#Chords with more distinct notes (i.e. max 3) are rewarded
        cost5= self.m.continuous_var_list(self.N, 0,100, "Distinct notes cost")
        for j in range(self.N):
            self.m.add_constraint(cost5[j]>=-1+
                                  self.m.sum( self.m.sum( (self.x[i,j]- self.x[k,j]==12) + (self.x[i,j]- self.x[k,j]==24) + (self.x[i,j]- self.x[k,j]==36) for i in range(3) ) for k in range(4) )
                                  ) 
        return cost5
    def soft_constraint_voice_crossing(self): # is a hard constraint
        pass
    
    def soft_constraint_voice_range(self, slb = [24,17, 10], sub = [33,23 ,21],weight=1):
        cost6= self.m.continuous_var_list(self.N, 0,100, "voice range cost")
        
        
        for j in range(self.N):
            cost6[j]>=self.m.sum( self.x[i,j] -sub[i-1]  for i in range(1,4))
            cost6[j]>=self.m.sum( slb[i-1]-self.x[i,j]  for i in range(1,4))
        return  cost6
                        
    def solve(self, log = True):
         
        recorder = ProgressDataRecorder()
        self.m.add_progress_listener(recorder)
        sol = self.m.solve(log_output = log)
        #print(sol.get_objective_values())  
        
        # best_bound = []
        # det_time = []
        # current_objective = []
        # for data in recorder.recorded:
        #     best_bound.append(data.best_bound)
        #     det_time.append(data.det_time/100)
           
           
        # current_objective.append(data.current_objective)
           
        # plt.plot(det_time, current_objective, label = "best integer", marker='s', linestyle = '--')
        # plt.plot(det_time, best_bound, label = "best node")
        # plt.legend()
        # plt.show()   
        
        
        print(sol)
        return sol
        
        
    #
