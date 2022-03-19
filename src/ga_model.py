import src.music_functions
import numpy as np
import random

# how does our model add value over EvoComposer?

## multiobjective evaluation function?
## 

# define population class for the basis of the GA
class Population:
    
    def __init__(self):
        self.population = []
        self.fronts = []
    
    def __len__(self):
        return len(self.population)
    
    def __iter__(self):
        return self.population.__iter__()
    
    def extend(self, new_individuals):
        self.population.extend(new_individuals)
    
    def append(self, new_individuals):
        self.population.append(new_individual)

# define individual class; in this case, an individual is a single chorale, harmonizd
class Individual(object):
    
    def __init__(self, musical_input, x = None, c = None):
        self.rank = None
        self.musical_input = musical_input
        self.crowding_distance = None
        self.domination_count = None
        self.dominated_solutions = None
        self.x = x # notes decision variables
        self.c = c # chords decision variables
        self.objectives = None
    
    '''
    edit this function because features ~= musical note outputs ???????
    '''
    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.features == other.features
        return False
    
    def dominates(self, other_individual):
        and_condition = True
        or_condition = False
        for first, second in zip(self.objectives, other_individual.objectives):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
        return (and_condition and or_condition)



# define GA model class with constraints for the specific problem
class ga_model:
    
    def __init__(self, musical_input, chord_vocab, max_generation, population_size, prob_crossover, prob_mutation, prob_tournament, hard_constraints_c, hard_constraints_x, soft_constraints):
        self.musical_input = musical_input # An instance of the class MusicalWorkInput
        self.chord_vocab = chord_vocab # assuming the chord vocab has the major/minor ones parsed accordingly
        self.N = self.musical_input.melody_len
        self.K = self.musical_input.key
        self.x = np.zeros((4, self.N), dtype=int)
        self.c = np.zeros((self.N,))
        self.max_generation = max_generation
        self.population_size = population_size
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
        self.prob_tournament = prob_tournament
        self.hard_constraints_c = hard_constraints_c # A dict with constraint names as key, and boolean values on whether to include that constraint in the model or not
        self.hard_constraints_x = hard_constraints_x
        self.soft_constraints = soft_constraints
        
        for j in range(self.N):
            self.x[0, j] = self.musical_input.melody[j]
                
        #Adding Constraints
        hard_constraints_c = {'first last chords': self.hard_constraint_first_last_chords,
                             'adjacent bar chords': self.hard_constraint_adjacent_bar_chords}
        
        hard_constraints_x = {'voice range': self.hard_constraint_voice_range,
                             'chord membership': self.hard_constraint_chord_membership,
                             'chord bass repetition': self.hard_constraint_chord_bass_repetition,
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
        
        #Objective Function
        ####### insert
    
    @staticmethod
    def hard_constraint_first_last_chords(musical_input, chord_vocab, c):
        if musical_input.tonality == 'major':
            for chord in chord_vocab:
                if chord.name == 'I':
                    n = chord.index
                    break
            c[0] = n
            c[musical_input.melody_len - 1] = n
        elif musical_input.tonality == 'minor':
            n1 = []
            for chord in chord_vocab:
                if chord.name == 'i':
                    n = chord.index
                    n1.append(chord.index)
                elif chord.name == 'I':
                    n = chord.index
                    n1.append(chord.index)
            c[0] = n
            c[musical_input.melody_len - 1] = n1
        return c
    
    @staticmethod
    def hard_constraint_adjacent_bar_chords(musical_input, chord_vocab, c):
        chord_list = [chord.name for chord in chord_vocab]
        for j in range(1, musical_input.melody_len - 1):
            if j % musical_input.meter == musical_input.first_on_beat:
                if c[j] == c[j - 1]:
                    if j != musical_input.melody_len - 2:
                        chord_choice = [x for i, x in enumerate(chord_list) if x != c[j - 1]]
                        c[j] = chord_choice[random.randrange(len(chord_choice))]
                    else:
                        chord_choice = [x for i, x in enumerate(chord_list) if x != c[j - 1] and x != c[j + 1]]
                        c[j] = chord_choice[random.randrange(len(chord_choice))]
        return c
    
    @staticmethod
    def hard_constraint_voice_range(gene_space = [], lb = [19, 12, 5], ub = [38, 28, 26]):
        # voice_ranges = {1: (19, 38), 2: (12, 28), 3: (5, 26)}
        for i in range(1,4):
            gene_space[i] = [lb[i - 1], ub[i - 1]]
        return gene_space
    
    @staticmethod
    def hard_constraint_chord_membership(musical_input, chord_vocab, lb = 5, ub = 60, c, x, gene_space = None): # All notes must belong to the same chord
        chord_vocab_ext = []
        for chord in chord_vocab:
            chord_vocab_ext.append(src.music_functions.extend_range(src.music_functions.transpose(chord.note_intervals, musical_input.key)))
        for j in range(musical_input.melody_len):
            for i in range(4):
                for chord, chord_ext in zip(chord_vocab, chord_vocab_ext):
                    if c[j] == chord.index:
                        idx_slice = len(chord_ext) // 3
                        note_choice = []
                        rand_inversion = random.randrange(3)
                        if rand_inversion == 0: # root inversion
                            if i == 1:
                                note_choice.extend(chord_ext[idx_slice*2:])
                            elif i == 2:
                                note_choice.extend(chord_ext[idx_slice:idx_slice*2])
                            elif i == 3:
                                note_choice.extend(chord_ext[:idx_slice])
                        elif rand_inversion == 1: # first inversion
                            random_roll = random.random() # to cater for any voicing above the bass
                            if random_roll > 0.5:
                                if i == 1:
                                    note_choice.extend(chord_ext[:idx_slice])
                                else i == 2:
                                    note_choice.extend(chord_ext[idx_slice*2:])
                            else:
                                if i == 1:
                                    note_choice.extend(chord_ext[idx_slice*2:])
                                else i == 2:
                                    note_choice.extend(chord_ext[:idx_slice])
                            if i == 3:
                                note_choice.extend(chord_ext[idx_slice:idx_slice*2])
                        elif rand_inversion == 2: # second inversion
                            random_roll = random.random() # to cater for any voicing above the bass
                            if random_roll > 0.5:
                                if i == 1:
                                    note_choice.extend(chord_ext[:idx_slice])
                                else i == 2:
                                    note_choice.extend(chord_ext[idx_slice:idx_slice*2])
                            else:
                                if i == 1:
                                    note_choice.extend(chord_ext[idx_slice:idx_slice*2])
                                else i == 2:
                                    note_choice.extend(chord_ext[:idx_slice])
                            if i == 3:
                                note_choice.extend(chord_ext[idx_slice*2:])
                        if gene_space != None:
                            filtered_note_choice = [note for note in note_choice if note >= gene_space[i][0] and note <= gene_space[i][1]]
                            x[i, j] = filtered_note_choice[random.randrange(len(filtered_note_choice))]
                        else:
                            x[i, j] = note_choice[random.randrange(len(note_choice))]
        return x
    
    @staticmethod
    def hard_constraint_chord_bass_repetition(musical_input, chord_vocab_ext, c, x):
        for j in range(musical_input.melody_len - 1):
            if c[j] == c[j + 1] and x[3, j] == x[3, j + 1]:
                chord_ext = chord_vocab_ext[c[j]]
                idx = chord_ext.index(x[3, j])
                if idx < 7:
                    chord_ext = chord_ext[:7].remove(x[3, j])
                elif idx < 14:
                    chord_ext = chord_ext[7:14].remove(x[3, j])
                else:
                    chord_ext = chord_ext[14:].remove(x[3, j])
                x[3, j] = chord_ext[random.randrange(len(chord_ext))]
        return x
    
    @staticmethod
    def hard_constraint_voice_crossing(musical_input, chord_vocab_ext, c, x):
        for i in range(1,3):
            for j in range(musical_input.melody_len):
                if x[i, j] < x[i + 1, j]:
                    chord_ext = chord_vocab_ext[c[j]]
                    idx = chord_ext.index(x[i, j])
                    if idx < 7:
                        chord_ext = chord_ext[:7]
                    elif idx < 14:
                        chord_ext = chord_ext[7:14]
                    else:
                        chord_ext = chord_ext[14:]
                    chord_ext = [note for note in chord_ext if note >= x[i+1, j]]
                    x[i, j] chord_ext[random.randrange(len(chord_ext))]
        return x
    
    @staticmethod
    def hard_constraint_parallel_movement(musical_input, x, disallowed_intervals = [7, 12]):
        for j in range(musical_input.melody_len - 1):
            for i1 in range(4):
                for i2 in range(4):
                    for interval in disallowed_intervals:
                        if (x[i1, j] - x[i2, j]) % 12 == interval
                            
                    
    
    # generate an individual when initializing population
    def generate_individual(self):
        individual = Individual(self.musical_input, self.x)
        chord_vocab_ext = []
        for chord in self.chord_vocab:
            chord_vocab_ext.append(src.music_functions.extend_range(src.music_functions.transpose(chord.note_intervals, self.K)))
        self.c = hard_constraint_first_last_chords(self.musical_input, self.chord_vocab, self.c)
        self.c = hard_constraint_adjacent_bar_chords(self.musical_input, self.chord_vocab, self.c)
        self.x = 
                # randomly generate chords and corresponding notes?????
                # then go through constraints because omg idk
                
        
        # get the tonality/key signature of the musical input
        # 1) ensure that the first and last chord is the tonal chord
        # 2) ensure that all notes must belong to a chord
        # soft
        
    # initialize population
    def initialize(self):
        population = Population()
        for _ in range(self.population_size):
            individual = self.problem
        return population

    # evaluate popultaion
    def evaluate_pop(self):
        return score
    
    # as per NSGA-II approach -> O(MN^2)
    def fast_nondominated_sort(self, population):
        return population
    
    # as per NSGA-II approach -> O(MN^2)
    def crowding_distance(self, front):
        #return population
                                      
    # select parents
    def select_parents(self):
        '''
        roulette wheel selection
        ensuring both elitism (important to preserve solutions having specific music harmonic and melodic properties) and diversity (allows avoidance of harmonic & melodic flattening; always obtaining new ideas during composition process)
        '''
        return parents #???
# crossover
'''
order crossover OX - better suited for cyclic permutations like in TSP
From parents P1 and P2:
- first, cutting sites i and j are randomly selected in P1. Then the substring P1(i) ... P1(j) is copied into C1(i) ... C1(j).
- finally, P2 is swept CIRCULARLY from j+1 onward to complete C1 with the missing nodes. C1 is also filled circularly from j+1.
- the other child C2 may be obtained by exchanging the roles of P1 and P2.
whole procedure can be implemented in O(n)
'''

# mutation

# fitness function