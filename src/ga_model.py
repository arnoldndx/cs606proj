import random
import numpy as np
import src.music_functions

# most hard constraints will be translated to high penalty scores
# single-objective genetic algorithm & fitness function

class Population:
    
    def __init__(self):
        self.population = []
        
    def __len__(self):
        return len(self.population)
    
    def __iter__(self):
        return self.population.__iter__()
    
    def extend(self, new_individuals):
        self.population.extend(new_individuals)
    
    def append(self, new_individuals):
        self.population.append(new_individual)
        
class Individual:
    
    def __init__(self, musical_input, x=None, c=None):
        self.rank = None
        self.musical_input = musical_input
        self.fitness_score = 0
        self.x = x # decision variable for notes
        self.c = c # decision variable for chords

class Constraints:
    
    def __init__(self, hard_constraints, soft_constraints):
        self.hard_constraints = hard_constraints
        self.soft_constraints = soft_constraints
        
        hard_constraints = {'musical input': self.hard_constraint_musical_input,
                            'voice range': self.hard_constraint_voice_range,
                            'chord membership': self.hard_constraint_chord_membership, # also in initialize_harmony()
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
        
class ga_model:
    
    def __init__(self, musical_input, chord_vocab, max_generation, population_size, crossover_prob, mutation_prob, tournament_prob, hard_constraints, soft_constraints):
        self.musical_input = musical_input # an instance of the class MusicalWorkInput
        self.chord_vocab = chord_vocab
        self.chord_vocab_ext = []
        self.N = self.musical_input.melody_len
        self.K = self.musical_input.key
        self.x = np.zeros((4, self.N), dtype=int)
        self.c = np.zeros((self.N,))
        self.max_generation = max_generation
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_prob = tournament_prob
        self.hard_constraints = hard_constraints
        self.soft_constraints = soft_constraints
        
        for chord in self.chord_vocab:
            chord_vocab_ext.append(src.music_functions.extend_range(src.music_functions.transpose(chord.note_intervals, self.K)))
            
        for j in range(self.N):
            self.x[0,j] = self.musical_input.melody[j]
    
    @staticmethod
    def initialize_chords(musical_input, chord_vocab, c):
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
        key = musical_input.key
        for idx in range(1, len(c)):
            chord_idx = random.randrange(len(chord_vocab))
            chord_name = chord_vocab[chord_idx].name
            c[idx] = chord
        return c
    
    @staticmethod
    def initialize_harmony(musical_input, chord_vocab, chord_vocab_ext, c, x, voice_range):
        for j in range(musical_input.melody_len):
            for i in range(1,4):
                for chord, chord_ext in zip(chord_vocab, chord_vocab_ext):
                    if c[j] == chord:
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
                        filtered_note_choice = [note for note in note_choice if note >= voice_range[i][0] and note <= voice_range[i][1]]
                        x[i, j] = filtered_note_choice[random.randrange(len(filtered_note_choice))]
        return x
    
    @staticmethod
    def fitness_calculation(x, c):
        
        return fitness
    
    def generate_individual(self):
        individual = Individual(self.musical_input, self.x)
        individual.c = initialize_chords(self.musical_input, self.chord_vocab, self.c)
        voice_range = hard_constraint_voice_range()
        individual.x = initialize_harmony(self.musical_input, self.chord_vocab, self.chord_vocab_ext, self.c, self.x, voice_range)
        individual.fitness_score = fitness_calculation(individual.x, individual.c)
        return individual
            
    def initialize_population(self):
        population = Population()
        for _ in range(self.population_size):
            individual = generate_individual()
            population.append(individual)
        return population