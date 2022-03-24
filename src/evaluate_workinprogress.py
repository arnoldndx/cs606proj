from src.music_functions import *
import pandas as pd
#%%
def evaluate_cost(list_x, list_c, key, tonality, meter = 4, first_on_beat = 0, mode="L",
                  chord_vocab = None,
                  chord_progression_penalties = None,
                  hard_constraints = None, hard_constraint_weight = 1000,
                  soft_constraint_weights = None): 
    '''
    Parameters
    ----------
    list_x : list
        4*N 2-dim list : 4 part notes
    list_c : list
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
        The default is "L".
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
    if chord_progression_penalties is None:
        penalties_chord_progression_major = pd.read_csv("../data/chord_progression_major.csv", header = 1, index_col = 0)
        penalties_chord_progression_minor = pd.read_csv("../data/chord_progression_minor.csv", header = 1, index_col = 0)
        penalties_chord_progression_major = dict(penalties_chord_progression_major.stack())
        penalties_chord_progression_minor = dict(penalties_chord_progression_minor.stack())
        chord_progression_costs = penalties_chord_progression_major if tonality=="major" else penalties_chord_progression_minor
    else:
        chord_progression_costs = chord_progression_penalties
        
    #Hard Constraint Weights
    if hard_constraints is None:
        hard_constraint_options = ['musical input', 'voice range', 'chord membership', 'first last chords',
                           'chord repetition', 'chord bass repetition', 'adjacent bar chords', 'voice crossing', 'parallel movement',
                          'chord spacing', 'incomplete chord', 'chord distribution']
        hard_constraint_w_weights = {x: hard_constraint_weight for x in hard_constraint_options}
    else:
        hard_constraint_w_weights = {k: hard_constraint_weight if v else 0 for k, v in hard_constraints.items()}
    
    #Soft Constraint Weights
    if soft_constraint_weights is None:
        weight_df = pd.read_csv("../data/soft_constraint_weights_temp.csv")
        soft_constraint_w_weights={}
        for _,name, w in weight_df.itertuples(): #name population is same as soft_constraint_options
            soft_constraint_w_weights[name]=float(w)
    else:
        soft_constraint_w_weights = soft_constraint_weights
    assert sum(v for v in soft_constraint_w_weights.values() if v > 0) == 100
           
    # Importing Chord Vocabulary
    # if tonality=="major":
    #     chord_df = pd.read_csv("../data/chord_vocabulary_major.csv", index_col = 0)
    # else:
    #     chord_df = pd.read_csv("../data/chord_vocabulary_minor.csv", index_col = 0)
    # chord_vocab = []
    # for name, note_intervals in chord_df.itertuples():
    #     chord_vocab.append(Chord(name, set(int(x) for x in note_intervals.split(','))))

#***************************************************************************************************************  
        
    def hard_constraint_voice_range(weight = hard_constraint_weight, lb = [19, 12, 5], ub = [38, 28, 26]):
        cost = 0
        for i in range(1,4):
            for j in range(N):
                cost += (x[i,j] >= lb[i-1]) + (x[i,j] <= ub[i-1])
        return weight * cost
    
    def hard_constraint_chord_membership(weight = hard_constraint_weight): #All notes must belong to the same chord
        cost = 0
        for j in range(N):
            notes = transpose(chord_vocab[c[j]].note_interval, n_semitones = key, mod = True, ascending = True)
            for i in range(4):
                cost += (x[i,j] % 12) not in notes
        return weight * cost
    
    def hard_constraint_first_last_chords(weight = hard_constraint_weight):
        cost = 0
        if tonality == "major":
            cost += c[0] != "I"
            cost += c[N-1] != "I"
        else:
            cost += c[0] != "i"
            cost += c[N-1] not in ["i", "I"]
        
        #First and last bass notes must be the tonic note
        cost += x[3,0] % 12 == K
        cost += x[3,self.N-1] % 12 == K
        
        return weight * cost

    def hard_constraint_chord_repetition(self):
        cost = 0
        for j in range(N-1):
            cost += c[j+1] != self.c[j]
        return weight * cost
    
    def hard_constraint_chord_bass_repetition(weight = hard_constraint_weight):
        cost = sum((c[j] == c[j+1]) > (x[3,j] != x[3,j+1]) for j in range(N-1)) 
        return weight * cost
    # def hard_constraint_adjacent_bar_chords(weight = hard_constraint_weight):
        # for j in range(1,N):
            # if j % musical_input.meter == musical_input.first_on_beat:
                # m.add_constraint(c[j] != c[j-1])
    def hard_constraint_adjacent_bar_chords(weight = hard_constraint_weight):
        cost = 0
        for j in range(1,N):
            if j % meter == first_on_beat:
                cost += (c[j] == c[j-1])
        return weight * cost
    
    def hard_constraint_voice_crossing(weight = hard_constraint_weight):

        return weight*sum(x[i,j] < x[i+1,j]  for i in range(3) for j in range(N))
    
    def hard_constraint_parallel_movement(weight = hard_constraint_weight, disallowed_intervals = [7, 12]):
        cost=0
        for j in range(N-1):
            for i in range(3):
                for k in range(i+1, 4):
                    for interval in disallowed_intervals:
                        cost+=( (x[i,j]-x[k,j] ==interval)> ( x[i,j+1]-x[k,j+1] !=interval) )
                        cost+=( (x[i,j]-x[k,j] ==interval+12)> ( x[i,j+1]-x[k,j+1] !=interval+12) )
                        cost+=( (x[i,j]-x[k,j] ==interval+24)> ( x[i,j+1]-x[k,j+1] !=interval+24) )
        return weight*cost
    def hard_constraint_chord_spacing(weight = hard_constraint_weight, max_spacing = [12, 12, 16]):
        cost=0
        for j in range(N):
            for i in range(3):
                cost+=((x[i,j] - x[i+1,j]) > max_spacing[i-1])
        return weight*cost        

#***************************************************************************************************************       
    def soft_constraint_chord_progression( weight=1):

        cost= weight*sum(chord_progression_costs.get((c[j],c[j+1]), 1000) for j in range(N-1))
        return cost
    def soft_constraint_leap_resolution(max_leap=10, weight=1): # leaps more than an interval of a major 6th should resolve in the opposite direction by stepwise motion(

        cost= weight*sum((1-(x[i,j] - x[i,j+1] <= max_leap) for i in range(1,4) for j in range(N-1))  )                
        return cost
    def soft_constraint_melodic_movement( weight=1):
        pass
    def soft_constraint_chord_repetition( weight=2):

        cost= weight*sum(c[j]==c[j+1] for j in range(N-1))
        return cost
    def soft_constraint_chord_bass_repetition( weight=1):    
        pass
    def soft_constraint_adjacent_bar_chords( weight=1):
        pass
    
    
    def soft_constraint_note_repetition( weight=2):
        cost= weight*sum(sum( (x[i,j]==x[i,j+1])*(x[i,j+1]==x[i,j+2]) 
                          for i in range(1,4))
                          for j in range(N-2))
        return cost                 

    def soft_constraint_parallel_movement( weight=1):# is a hard constraint
        pass
    
    def soft_constraint_voice_overlap(weight=1):

        return weight*sum((x[i,j+1] <= x[i+1,j]-1) for i in range(3) for j in range(N-1))
    def soft_constraint_chord_spacing( weight=1):
        pass
    
    def soft_constraint_distinct_notes(weight=1):#Chords with more distinct notes (i.e. max 3) are rewarded
                                 
        cost=weight*sum(sum( sum( (x[i,j]- x[k,j]==12) + (x[i,j]- x[k,j]==24) + (x[i,j]- x[k,j]==36) for i in range(3) ) for k in range(4) ) for j in range(N))
        return cost
    def soft_constraint_voice_crossing( weight=1): # is a hard constraint
        pass
    
    def soft_constraint_voice_range( slb = [24,17, 10], sub = [33,23 ,21],weight=1):
        return weight*sum( max(x[i,j] -sub[i-1],0) +  max(slb[i-1]-x[i,j],0) for i in range(1,4) for j in range(N))

    
    
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
        for k, v in self.hard_constraints.items():
            if v:
                hard_constraints[k]()
        for k, v in self.soft_constraints_weights.items():
            if v > 0:
                soft_constraints[k]()
    
    #Compiling Result
    result = [hard_constraints[k]() if v > 0 for k, v in hard_constraint_w_weights.items()]
    result.extend(soft_constraints[k]() if v > 0 for k, v in soft_constraint_w_weights.items())
    
    if mode == "D":
        return result
    else:
        result_list = list(result.values())
        if mode == "L":
            return result
        else:
            return sum(result)

#cost=src.evaluate.evaluate_cost(solution[:-1],solution[-1] ,"major", mode="L")   
#print(cost)            