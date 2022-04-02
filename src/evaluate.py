from src.music_functions import *
from src.chord import *
import pandas as pd

def evaluate_cost(list_x, list_c, key, tonality, meter = 4, first_on_beat = 0, mode = "D",
                  chord_vocab = None,
                  chord_progression_penalties = None,
                  hard_constraints = None, hard_constraint_weight = 1000,
                  soft_constraint_weights = None): 
    '''
    Parameters
    ----------
    list_x : list of int
        4*N 2-dim list : 4 part notes
    list_c : list of int
        N 1-dim list : chords enumerated
    key : int from 0 to 11
        0 indicates "C" key, 1, indicates "C#" key, 2 indicates "D" key, ... 11 indicates "B" key
    tonality : TYPE
        "major" or "minor"
    meter :  optional
         The default is 4.
    first_on_beat : 0 to 3, optional
        The default is 0.
    mode : string, optional
        The default is "D".
        If "L": return will be a list of sum-of-cost for each hard/soft constraint
        If "D": return will be a dictionary of sum-of-cost for each hard/soft constraint
        Otherwise, : return will be one number,sum of all costs for all constraints
    chord_vocab : list of Chord objects or None
        If None, it will be read from csv.
    chord_progression_penalties : dict
        Dict key is a tuple of the form (chord1.name, chord2.name), dict value is the penalty for transiting from chord1 to chord2
    hard_constraints : dict or None. If None, it is assumed all hard_constraints are implemented.
        Dict key is the name of hard constraint. Dict value is a bool, whether or not the hard constraint is used.
    hard_constraint_weight : float/int
        How much to weight hard constraint costs. For comparison, by convention the total weight of soft constraints is 100.
    soft_constraint_weights : dict or None. If none, weights will be read from csv.
        Dict key is the name of soft constraint. Dict value is an int from 1 to 99 inclusive, indicating the weight of that soft constraint.
    
    Returns
    list or number
    
    #!!! Weights are hard coded, hard constraints are given weight 1000
    #!!! incomplete hard / soft constaints inplemeted
    '''
    N = len(list_c)
    assert len(list_x) == 4
    assert len(list_x[0]) == N
    c = list_c.copy()
    x = {(i,j): list_x[i][j] for i in range(4) for j in range(N)} 

    #Chord Vocab
    #To avoid reading data for every evaluate function call, the chord_vocab is provided as an optional argument
    if chord_vocab is None:
        chord_df_major = pd.read_csv("../data/chord_vocabulary_major.csv", index_col = 0)
        chord_df_minor = pd.read_csv("../data/chord_vocabulary_minor.csv", index_col = 0)
        chord_vocab_major, chord_vocab_minor = [], []
        for index, name, note_intervals in chord_df_major.itertuples():
            chord_vocab_major.append(Chord(index, name, [int(x) for x in note_intervals.split(',')]))
        for index, name, note_intervals in chord_df_minor.itertuples():
            chord_vocab_minor.append(Chord(index, name, [int(x) for x in note_intervals.split(',')]))
        chord_vocab = chord_vocab_major if tonality == "major" else chord_vocab_minor
    
    #Chord Progression Costs
    #To avoid reading data for every evaluate function call, the chord progression costs is provided as an optional argument
    if chord_progression_penalties is None:
        penalties_chord_progression_major = pd.read_csv("../data/chord_progression_major.csv", header = 1, index_col = 0)
        penalties_chord_progression_minor = pd.read_csv("../data/chord_progression_minor.csv", header = 1, index_col = 0)
        penalties_chord_progression_major = dict(penalties_chord_progression_major.stack())
        penalties_chord_progression_minor = dict(penalties_chord_progression_minor.stack())
        chord_progression_costs = penalties_chord_progression_major if tonality=="major" else penalties_chord_progression_minor
    else:
        chord_progression_costs = chord_progression_penalties
        
    #Hard Constraint Weights
    hard_constraint_options = ['voice range', 'chord membership', 'first last chords',
                               'chord repetition', 'chord bass repetition', 'adjacent bar chords', 'voice crossing', 'parallel movement',
                               'chord spacing', 'incomplete chord', 'chord distribution']
    if hard_constraints is None: #Activate all hard constraints
        hard_constraint_w_weights = {x: hard_constraint_weight for x in hard_constraint_options}
    else:
        hard_constraint_w_weights = {x: hard_constraint_weight if hard_constraints[x] else 0 for x in hard_constraint_options}
    
    #Soft Constraint Weights
    #Provide argument to avoid reading daa for every evaluate function call
    if soft_constraint_weights is None:
        weight_df = pd.read_csv("../data/soft_constraint_weights_temp.csv")
        soft_constraint_w_weights={}
        for _,name, w in weight_df.itertuples(): #name population is same as soft_constraint_options
            soft_constraint_w_weights[name]=float(w)
    else:
        soft_constraint_w_weights = soft_constraint_weights
    assert sum(v for v in soft_constraint_w_weights.values() if v > 0) == 100

#***************************************************************************************************************  

    ### Common Constraints

    def chord_repetition(weight):
        assert weight > 0
        cost = 0
        for j in range(N-1):
            cost += c[j+1] == c[j]
        return weight * cost
    
    def chord_bass_repetition(weight):
        assert weight > 0
        cost = sum(c[j] == c[j+1] and x[3,j] == x[3,j+1] for j in range(N-1))
        return weight * cost
    
    def adjacent_bar_chords(weight):
        assert weight > 0
        cost = 0
        for j in range(1,N):
            if j % meter == first_on_beat:
                cost += (c[j] == c[j-1])
        return weight * cost
    
    def voice_crossing(weight):
        assert weight > 0
        cost = sum(x[i,j] < x[i+1,j] for i in range(3) for j in range(N))
        return weight * cost
    
    def parallel_movement(weight, intervals = [0, 7]):
        assert weight > 0
        intervals_ext = []
        for k in range(-1, 3):
            if k != 0:
                intervals_ext.extend([interval + 12 * k for interval in intervals])
        cost = 0
        for j in range(N-1):
            for i in range(3):
                for k in range(i+1, 4):
                    for interval in intervals_ext:
                        cost += x[i,j]-x[k,j] == interval and x[i,j] != x[i,j+1] and x[i,j+1]-x[k,j+1] == interval
        return weight * cost
    
    def chord_spacing(weight, max_spacing = [12, 12, 16]):
        assert weight > 0
        cost = 0
        for j in range(N):
            for i in range(3):
                cost += ((x[i,j] - x[i+1,j]) > max_spacing[i])
        return weight * cost
    
    def incomplete_chord(weight): #The 4 voices must fully cover the 3 notes in a chord
        assert weight > 0
        cost = 0
        for j in range(N):
            cost += any([(note + key) % 12 not in [x[0,j]%12, x[1,j]%12, x[2,j]%12, x[3,j]%12] for note in chord_vocab[c[j]].note_intervals]) #Evaluates to True if any note in the assigned chord c[j] is not covered by any of the 4 voices
        
        return weight * cost
    
    def chord_distribution(weight):
        #Distance between adjacent lower voices must not be less than distance between adjacent higher voices.
        assert weight > 0
        cost = 0
        for j in range(N):
            for i in range(2):
                cost += x[i,j] - x[i+1,j] > x[i+1,j] - x[i+2,j]
        
        return weight * cost

#***************************************************************************************************************  

    ### Hard Constraints

    def hard_constraint_voice_range(weight = hard_constraint_weight, lb = [17, 12, 4], ub = [41, 36, 28]):
        assert weight > 0
        cost = 0
        for i in range(1,4):
            for j in range(N):
                cost += (x[i,j] < lb[i-1]) + (x[i,j] > ub[i-1])
        return weight * cost
    
    def hard_constraint_chord_membership(weight = hard_constraint_weight): #All notes must belong to the same chord
        assert weight > 0
        cost = 0
        for j in range(N):
            notes = transpose(chord_vocab[c[j]].note_intervals, n_semitones = key, mod = True, ascending = True)
            for i in range(4):
                cost += (x[i,j] % 12) not in notes
        return weight * cost
    
    def hard_constraint_first_last_chords(weight = hard_constraint_weight):
        assert weight > 0
        cost = 0
        if tonality == "major":
            cost += chord_vocab[c[0]].name != "I"
            cost += chord_vocab[c[N-1]].name != "I"
        else:
            cost += chord_vocab[c[0]].name != "i"
            cost += chord_vocab[c[N-1]].name not in ["i", "I"]
        
        #First and last bass notes must be the tonic note
        cost += x[3,0] % 12 != key
        cost += x[3,N-1] % 12 != key
        
        return weight * cost

    def hard_constraint_chord_repetition(weight = hard_constraint_weight):
        return chord_repetition(weight)
    
    def hard_constraint_chord_bass_repetition(weight = hard_constraint_weight):
        return chord_bass_repetition(weight)

    def hard_constraint_adjacent_bar_chords(weight = hard_constraint_weight):
        return adjacent_bar_chords(weight)
    
    def hard_constraint_voice_crossing(weight = hard_constraint_weight):
        return voice_crossing(weight)
    
    def hard_constraint_parallel_movement(weight = hard_constraint_weight, disallowed_intervals = [0, 7]):
        return parallel_movement(weight, intervals = disallowed_intervals)

    def hard_constraint_chord_spacing(weight = hard_constraint_weight, max_spacing = [12, 12, 16]):
        return chord_spacing(weight, max_spacing)
    
    def hard_constraint_incomplete_chord(weight = hard_constraint_weight): #The 4 voices must fully cover the 3 notes in a chord
        return incomplete_chord(weight)
    
    def hard_constraint_chord_distribution(weight = hard_constraint_weight): #Activating this may result in no feasible solution.
        return chord_distribution(weight)

#***************************************************************************************************************       

    ### Soft Constraints

    def soft_constraint_chord_progression(weight = soft_constraint_w_weights['chord progression'], progression_costs = chord_progression_costs):
        assert weight > 0
        cost = sum(chord_progression_costs[chord_vocab[c[j]].name, chord_vocab[c[j+1]].name] for j in range(N-1))
        return weight * cost

    def soft_constraint_chord_repetition(weight = soft_constraint_w_weights['chord repetition']):
        return chord_repetition(weight)
    
    def soft_constraint_chord_bass_repetition(weight = soft_constraint_w_weights['chord bass repetition']):
        return chord_bass_repetition(weight)
    
    def soft_constraint_leap_resolution(weight = soft_constraint_w_weights['leap resolution'], max_leap = 10): # leaps more than an interval of a major 6th should resolve in the opposite direction by stepwise motion(
        assert weight > 0
        cost = sum((x[i,j] - x[i,j+1] > max_leap for i in range(1,4) for j in range(N-1)))
        return weight * cost
    
    def soft_constraint_melodic_movement(weight = soft_constraint_w_weights['melodic movement']):
        #This is continuous method. If binary method, leap_interval = {1: 12, 2: 12, 3: 16}. Incur cost if leap is more than leap interval.
        assert weight > 0
        cost = sum(abs(x[i,j+1] - x[i,j])/12 for i in range(1,4) for j in range(N-1))
        return weight * cost

    def soft_constraint_note_repetition(weight = soft_constraint_w_weights['note repetition']):
        assert weight > 0
        cost = sum(sum( (x[i,j]==x[i,j+1]) and (x[i,j+1]==x[i,j+2]) 
                          for i in range(1,4))
                          for j in range(N-2))
        return weight * cost
    
    def soft_constraint_parallel_movement(weight = soft_constraint_w_weights['parallel movement'], discouraged_intervals = [0, 7]):
        return parallel_movement(weight, intervals = discouraged_intervals)
    
    def soft_constraint_voice_overlap(weight = soft_constraint_w_weights['voice overlap']):
        assert weight > 0
        cost = sum(x[i,j] > x[i+1,j] and x[i+1,j+1] > x[i,j] for i in range(3) for j in range(N-1))
        return weight * cost
    
    def soft_constraint_adjacent_bar_chords(weight = soft_constraint_w_weights['adjacent bar chords']):
        return adjacent_bar_chords(weight)

    def soft_constraint_chord_spacing(weight = soft_constraint_w_weights['chord spacing'], max_spacing = [12, 12, 16]):
        return chord_spacing(weight, max_spacing)
    
    def soft_constraint_distinct_notes(weight = soft_constraint_w_weights['distinct notes']): #Chords with 3 notes repeated are penalised #Chords with repeated 3rd are penalised
        assert weight > 0
        cost = 0
        for j in range(N):
            note = chord_vocab[c[j]].note_intervals[1]
            note = (note + key) % 12
            for i1 in range(4):
                for i2 in range(4):
                    if i1 < i2:
                        for i3 in range(4):
                            if i2 < i3:
                                cost += x[i1,j]%12 == x[i2,j]%12 and x[i2,j]%12 == x[i3,j]%12 #Penalising 3 same notes in a chord
                        cost += x[i1,j]%12 == note and x[i2,j]%12 == note #Doubled 3rd note
        return weight * cost
    
    def soft_constraint_incomplete_chord(weight = soft_constraint_w_weights['incomplete chord']):
        return incomplete_chord(weight)
    
    def soft_constraint_voice_crossing(weight = soft_constraint_w_weights['voice crossing']):
        return voice_crossing(weight)
    
    def soft_constraint_voice_range(weight = soft_constraint_w_weights['voice range'], lb = [17, 12, 4], ub = [41, 36, 28], threshold = 2):
        assert weight > 0
        cost = sum((x[i,j] < (lb[i-1] + threshold)) + (x[i,j] > (ub[i-1] - threshold)) for i in range(1,4) for j in range(N))
        return weight * cost

    def soft_constraint_second_inversion(weight = soft_constraint_w_weights['second inversion']):
        assert weight > 0
        cost = sum(x[3,j] % 12 == (chord_vocab[c[j]].note_intervals[-1] + key) % 12 for j in range(N))
        return weight * cost
    
    def soft_constraint_first_inversion(weight = soft_constraint_w_weights['first inversion']):
        assert weight > 0
        cost = sum(x[3,j] % 12 == (chord_vocab[c[j]].note_intervals[1] + key) % 12 for j in range(N))
        return weight * cost
    
    def soft_constraint_chord_distribution(weight = soft_constraint_w_weights['chord distribution']):
        return chord_distribution(weight)

#***************************************************************************************************************       

    #All Constraint Options
    hard_constraints = {'voice range': hard_constraint_voice_range,
                        'chord membership': hard_constraint_chord_membership,
                        'first last chords': hard_constraint_first_last_chords,
                        'chord repetition': hard_constraint_chord_repetition,
                        'chord bass repetition': hard_constraint_chord_bass_repetition,
                        'adjacent bar chords': hard_constraint_adjacent_bar_chords,
                        'voice crossing': hard_constraint_voice_crossing,
                        'parallel movement': hard_constraint_parallel_movement,
                        'chord spacing': hard_constraint_chord_spacing,
                        'incomplete chord': hard_constraint_incomplete_chord,
                        'chord distribution': hard_constraint_chord_distribution}

    soft_constraints = {'chord progression': soft_constraint_chord_progression,
                        'chord repetition': soft_constraint_chord_repetition,
                        'chord bass repetition': soft_constraint_chord_bass_repetition,
                        'leap resolution': soft_constraint_leap_resolution,
                        'melodic movement': soft_constraint_melodic_movement,
                        'note repetition': soft_constraint_note_repetition,
                        'parallel movement': soft_constraint_parallel_movement,
                        'voice overlap': soft_constraint_voice_overlap,
                        'adjacent bar chords': soft_constraint_adjacent_bar_chords,
                        'chord spacing': soft_constraint_chord_spacing,
                        'distinct notes': soft_constraint_distinct_notes,
                        'incomplete chord': soft_constraint_incomplete_chord,
                        'voice crossing': soft_constraint_voice_crossing,
                        'voice range': soft_constraint_voice_range,
                        'second inversion': soft_constraint_second_inversion,
                        'first inversion': soft_constraint_first_inversion,
                        'chord distribution': soft_constraint_chord_distribution}

#***************************************************************************************************************       
    
    #Compiling Result
    result = {'hard constraint {}'.format(k): hard_constraints[k]() if v > 0 else 0 for k, v in hard_constraint_w_weights.items()}
    for k, v in soft_constraint_w_weights.items():
        if v > 0:
            result['soft constraint {}'.format(k)] = soft_constraints[k]()
        else:
            result['soft constraint {}'.format(k)] = 0
    
    if mode == "D":
        return result
    else:
        result_list = list(result.values())
        if mode == "L":
            return result_list
        else:
            return sum(result_list)