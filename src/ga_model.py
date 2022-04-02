import timeit
import copy
import random
import numpy as np
import pandas as pd
import src.music_functions
import src.evaluate_ga

# most hard constraints will be translated to high penalty scores
# single-objective genetic algorithm & fitness function

class Population:
    
    def __init__(self):
        self.population = []
        
    def __len__(self):
        return len(self.population)
    
    def _extend(self, new_individuals):
        self.population.extend(new_individuals)
    
    def _append(self, new_individuals):
        self.population.append(new_individuals)
    
    def _sort(self):
        assert len(self.population) > 0
        dct = {}
        x = 0
        for individual in self.population:
            dct[x] = individual.overall_fitness_score
            x += 1
        dct_sorted = dict(sorted(dct.items(), key=lambda item: item[1]))
        population_sorted = [] * len(self.population)
        for k, v in dct_sorted.items():
            population_sorted.append(self.population[k])
        self.population = population_sorted
    
    def evaluate_population(self):
        for individual in self.population:
            individual.overall_fitness_score, individual.gene_fitness_score = individual.calculate_overall_fitness()
    
    def sum_of_fitness(self):
        total_score = 0
        for individual in self.population:
            total_score += individual.overall_fitness_score
        return total_score
    
    def fitness_probability(self):
        probability_list = [0] * self.__len__()
        total_score = self.sum_of_fitness()
        for idx in range(self.__len__()):
            probability_list[idx] = self.population[idx].overall_fitness_score / total_score
        return probability_list
    
    #def calculate_threshold(self):
    #    score = 0
    #    for individual in self.population:
    #        score += individual.overall_fitness_score
    #    threshold = score / self.__len__()
    #    return threshold

class Individual:
    
    def __init__(self, musical_input, x=None, c=None, chord_vocab=None, chord_progression_penalties=None, hard_constraints=None, soft_constraint_weights=None):
        self.rank = None
        self.musical_input = musical_input
        self.overall_fitness_score = int(0)
        self.gene_fitness_score = [0] * self.musical_input.melody_len
        self.individual_len = self.musical_input.melody_len
        self.x = x # decision variable for notes
        self.c = c # decision variable for chords
        self.chord_vocab = chord_vocab
        self.chord_progression_penalties = chord_progression_penalties
        self.hard_constraints = hard_constraints
        self.soft_constraint_weights = soft_constraint_weights
    
    def calculate_overall_fitness(self):
        chord_list = [c.index for c in self.c]
        score_list = src.evaluate.evaluate_cost(
            list_x=self.x,
            list_c=chord_list,
            key=self.musical_input.key,
            tonality=self.musical_input.tonality,
            meter=self.musical_input.meter,
            first_on_beat=self.musical_input.first_on_beat,
            mode='GA',
            chord_vocab=self.chord_vocab,
            chord_progression_penalties=self.chord_progression_penalties,
            hard_constraints=self.hard_constraints,
            hard_constraint_weight=1000,
            soft_constraint_weights=self.soft_constraint_weights
        )
        score_list = [score / self.individual_len for score in score_list]
        overall_score = sum(x for x in score_list)
        return overall_score, score_list
    
    def crossover_points(self):
        i, j = 0, 0
        score_list = copy.deepcopy(self.gene_fitness_score)
        score_list_pair = [0] * (self.individual_len - 1)
        for idx in range(self.individual_len - 1):
            score_list_pair[idx] = score_list[idx+1] + score_list[idx]
        score_list_pair_tmp = copy.deepcopy(score_list_pair)
        for count in range(0,2):
            max_val = max(score_list_pair)
            if count == 0:
                i = score_list_pair.index(max_val)
                if len(score_list_pair) <= 3:
                    score_list_pair.remove(max_val)
                else:
                    if i == 0:
                        i_r = score_list_pair[i+1]
                        score_list_pair.remove(i_r)
                    elif i == len(score_list_pair) - 1:
                        i_l = score_list_pair[i-1]
                        score_list_pair.remove(i_l)
                    else:
                        i_l, i_r= score_list_pair[i-1], score_list_pair[i+1]
                        score_list_pair.remove(i_l)
                        score_list_pair.remove(i_r)
                    score_list_pair.remove(max_val)
            else:
                j = score_list_pair_tmp.index(max_val)
        if i > j:
            return j, i
        else:
            return i, j
    
    def initialize_harmony(self, chord_vocab_ext, voice_range):
        '''
        initialize the 3 voices' notes for the first iteration
        '''
        c = copy.deepcopy(self.c)
        x = copy.deepcopy(self.x)
        for j in range(self.individual_len):
            for i in range(1,4):
                for chord, chord_ext in zip(self.chord_vocab, chord_vocab_ext):
                    if c[j].index == chord.index:
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
                                elif i == 2:
                                    note_choice.extend(chord_ext[idx_slice*2:])
                            else:
                                if i == 1:
                                    note_choice.extend(chord_ext[idx_slice*2:])
                                elif i == 2:
                                    note_choice.extend(chord_ext[:idx_slice])
                            if i == 3:
                                note_choice.extend(chord_ext[idx_slice:idx_slice*2])
                        elif rand_inversion == 2: # second inversion
                            random_roll = random.random() # to cater for any voicing above the bass
                            if random_roll > 0.5:
                                if i == 1:
                                    note_choice.extend(chord_ext[:idx_slice])
                                elif i == 2:
                                    note_choice.extend(chord_ext[idx_slice:idx_slice*2])
                            else:
                                if i == 1:
                                    note_choice.extend(chord_ext[idx_slice:idx_slice*2])
                                elif i == 2:
                                    note_choice.extend(chord_ext[:idx_slice])
                            if i == 3:
                                note_choice.extend(chord_ext[idx_slice*2:])
                        filtered_note_choice = [note for note in note_choice if note >= voice_range[i-1][0] and note <= voice_range[i-1][1]]
                        x[i, j] = filtered_note_choice[random.randrange(len(filtered_note_choice))]
        return x
    
    def initialize_chords(self):
        '''
        initialize the chords for the first iteration
        '''
        c = [0] * self.individual_len
        if self.musical_input.tonality == 'major':
            for chord in self.chord_vocab:
                if chord.name == 'I':
                    n = chord
                    break
            c[0] = n
            c[self.individual_len - 1] = n
        elif self.musical_input.tonality == 'minor':
            n1 = []
            for chord in self.chord_vocab:
                if chord.name == 'i':
                    n = chord
                    n1.append(chord)
                elif chord.name == 'I':
                    n = chord
                    n1.append(chord)
            c[0] = n
            c[self.musical_input.melody_len - 1] = n
        key = self.musical_input.key
        for idx in range(1, len(c)-1):
            chord_idx = random.randrange(len(self.chord_vocab))
            chrd = self.chord_vocab[chord_idx]
            c[idx] = chrd
        return c
    
    def calculate_threshold(self):
        #threshold = 0
        return sum(self.gene_fitness_score) / self.individual_len
        #return threshold
    
class GAmodel:
    
    def __init__(self, musical_input, chord_vocab, max_generation, population_size, hard_constraints, soft_constraint_w_weights, chord_progression_penalties, mutation_probability):
        self.musical_input = musical_input # an instance of the class MusicalWorkInput
        self.chord_vocab = chord_vocab
        self.chord_vocab_ext = []
        self.N = self.musical_input.melody_len
        self.K = self.musical_input.key
        self.x = np.zeros((4, self.N))
        self.c = [0] * self.N
        self.max_generation = max_generation
        self.population_size = population_size
        self.hard_constraints = hard_constraints
        self.soft_constraint_w_weights = soft_constraint_w_weights
        self.chord_progression_penalties = chord_progression_penalties
        self.mutation_probability = mutation_probability
        self.voice_range_list = np.zeros((3, 2))

        for chord in self.chord_vocab:
            self.chord_vocab_ext.append(src.music_functions.extend_range(src.music_functions.transpose(chord.note_intervals, self.K)))
            
        for j in range(self.N):
            self.x[0,j] = self.musical_input.melody[j]
    
        lb = [19, 12, 5]
        ub = [38, 28, 26]
        #voice_ranges = {1: (19, 38), 2: (12, 28), 3: (5, 26)}
        for i in range(0,3):
            self.voice_range_list[i][0] = lb[i]
            self.voice_range_list[i][1] = ub[i]

    def row_col(self, lst, idx):
        res = []
        for x in range(len(lst)):
            res.append(lst[x][idx])
        return res
    
    def generate_individual(self):
        '''
        generate an individual for the first iteration
        '''

        individual = Individual(self.musical_input, self.x, None, self.chord_vocab, self.chord_progression_penalties, self.hard_constraints, self.soft_constraint_w_weights)
        individual.c = individual.initialize_chords()
        individual.x = individual.initialize_harmony(self.chord_vocab_ext, self.voice_range_list)
        individual.overall_fitness_score, individual.gene_fitness_score = individual.calculate_overall_fitness()
        return individual
    
    def initialize_population(self):
        '''
        initialize population for first iteration
        '''
    
        population = Population()
        tmp = [copy.deepcopy(self.generate_individual()) for i in range(self.population_size)]
        for t in tmp:
            population._append(t)
        return population
    
    def roulette_wheel_selection(self, population, pool_size):
        # function assumes that population at input is already sorted
        population_copy = copy.deepcopy(population)
        mating_pool = Population()
        for idx in range(pool_size):
            total_score = population_copy.sum_of_fitness()
            x = random.random()
            prob_list = copy.deepcopy(population_copy.fitness_probability())
            for idx in range(len(prob_list)):
                if x > prob_list[idx]:
                    indv = population_copy.population[idx]
                    mating_pool._append(indv)
                    population_copy.population.remove(indv)
                    break
        if mating_pool.__len__() < pool_size:
            diff = pool_size - mating_pool.__len__()
            for idx in range(diff):
                indv = population_copy.population[random.randrange(population_copy.__len__())]
                mating_pool._append(indv)
                population_copy.population.remove(indv)
        return mating_pool        
    
    def crossover(self, parent_1, mating_pool):
        # multi-point crossover
        '''
        multi-point crossover. in each individual, find the points whereby the pairs have the lowest fitness scores aka highest penalty scores. then scan through the rest of the mating pool and find the individual with the highest scores at the corresponding indexes.
        apply a multipoint crossover of the individual's gene with the other individual. that is the new child.
        '''
        
        i, j = parent_1.crossover_points()
        fitness_i, fitness_j = parent_1.gene_fitness_score[i], parent_1.gene_fitness_score[j]
        average_fitness = (fitness_i + fitness_j) / 2
        min_average, min_parent = 1e9, None
        for individual in mating_pool.population:
            if individual != parent_1:
                i_1, j_1 = individual.gene_fitness_score[i], individual.gene_fitness_score[j]
                curr_average = (i_1 + j_1) / 2
                min_average = min(min_average, curr_average)
                if min_average == curr_average:
                    min_parent = individual
        child = copy.deepcopy(parent_1)
        child.x = np.concatenate((parent_1.x[:,:i], min_parent.x[:,i:j], parent_1.x[:,j:]), axis=1)
        return child
    
    def mutation(self, individual, threshold):
        '''
        adaptive mutation. take the fitness score of each gene and check if they are above or below the threshold, and set the mutation rate accordingly.
        based on the mutation rate, check if the gene has to be mutated.
        either go through the entire indiviudal or hit the max number of mutations made, whichever hits earlier.
        '''
        
        chord_list = [chord.index for chord in self.chord_vocab]
        length = individual.individual_len
        low_mutation_rate, high_mutation_rate = self.mutation_probability[0], self.mutation_probability[1]
        c = copy.deepcopy(individual.c)
        x = copy.deepcopy(individual.x)
        gene_score = copy.deepcopy(individual.gene_fitness_score)
        for idx in range(individual.individual_len):
            if gene_score[idx] <= threshold:
                mutation_rate = low_mutation_rate
            else:
                mutation_rate = high_mutation_rate
            if random.random() <= mutation_rate:
                if idx == 0:
                    c_i, x_i = c[-1].index, self.row_col(x, idx)
                elif idx == individual.individual_len-1:
                    c_i, x_i = c[0].index, self.row_col(x, idx)
                else:
                    c_i, x_i = chord_list[random.randrange(len(chord_list))], self.row_col(x, idx)
                chord_tones = self.chord_vocab_ext[c_i]
                for val in x_i:
                    if val in chord_tones:
                        chord_tones.remove(val)
                idx_slice = len(chord_tones) // 3
                rand_inversion = random.randrange(3)
                note_choice = []
                for i in range(1,4):
                    if rand_inversion == 0: # root inversion
                        if i == 1:
                            note_choice.extend(chord_tones[idx_slice*2:])
                        elif i == 2:
                            note_choice.extend(chord_tones[idx_slice:idx_slice*2])
                        elif i == 3:
                            note_choice.extend(chord_tones[:idx_slice])
                    elif rand_inversion == 1: # first inversion
                        random_roll = random.random() # to cater for any voicing above the bass
                        if random_roll > 0.5:
                            if i == 1:
                                note_choice.extend(chord_tones[:idx_slice])
                            elif i == 2:
                                note_choice.extend(chord_tones[idx_slice*2:])
                        else:
                            if i == 1:
                                note_choice.extend(chord_tones[idx_slice*2:])
                            elif i == 2:
                                note_choice.extend(chord_tones[:idx_slice])
                        if i == 3:
                            note_choice.extend(chord_tones[idx_slice:idx_slice*2])
                    elif rand_inversion == 2: # second inversion
                        random_roll = random.random() # to cater for any voicing above the bass
                        if random_roll > 0.5:
                            if i == 1:
                                note_choice.extend(chord_tones[:idx_slice])
                            elif i == 2:
                                note_choice.extend(chord_tones[idx_slice:idx_slice*2])
                        else:
                            if i == 1:
                                note_choice.extend(chord_tones[idx_slice:idx_slice*2])
                            elif i == 2:
                                note_choice.extend(chord_tones[:idx_slice])
                        if i == 3:
                            note_choice.extend(chord_tones[idx_slice*2:])
                    filtered_note_choice = [note for note in note_choice if note >= self.voice_range_list[i-1][0] and note <= self.voice_range_list[i-1][1]]
                if len(filtered_note_choice) > 0:
                    x[i, idx] = filtered_note_choice[random.randrange(len(filtered_note_choice))]
                    c[idx] = self.chord_vocab[c_i]
        return x,c
    
    def solve(self):
        population = self.initialize_population()
        population.evaluate_population()
        population._sort() # sort the population in descending order of fitness score
        progress_array = []
        for _ in range(self.max_generation):
            start_time = timeit.default_timer()
            slicer = int(self.population_size * 0.1)
            if slicer % 2 != 0:
                slicer += 1
            new_population = Population()
            new_population._extend(population.population[:slicer])
            pool_size = self.population_size - slicer
            mating_pool = self.roulette_wheel_selection(population, pool_size)
            for x in range(0, mating_pool.__len__()):
                parent_1 = mating_pool.population[x]
                child_1 = self.crossover(parent_1, mating_pool)
                threshold = child_1.calculate_threshold()
                child_1.x, child_1.c = self.mutation(child_1, threshold)
                new_population._append(child_1)
            population = None
            population = copy.deepcopy(new_population)
            new_population = None
            population.evaluate_population()
            population._sort() # sort the population in descending order of fitness score
            end_time = timeit.default_timer()
            if _ > 0:
                progress_array.append((round(end_time - start_time + progress_array[-1][0],2), round(population.population[0].overall_fitness_score, 3)))
            else:
                progress_array.append((round(end_time - start_time,2), round(population.population[0].overall_fitness_score, 3)))
            #print(_, progress_array[-1])
        
        best_individual = population.population[0]
        best_solution = copy.deepcopy(best_individual.x)
        best_solution = np.vstack((best_solution, np.array(([chord.index for chord in best_individual.c])))).astype(int)
        best_solution = pd.DataFrame(best_solution)
        midi_array = copy.deepcopy(best_individual.x).astype(int)
        
        return best_solution, midi_array, progress_array