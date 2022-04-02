import pandas as pd

def infer_onset(notes):
    '''
    Function to infer onset of a given melody

    Parameters
    ----------
    notes : list of int
        A list of int corresponding to the pitch of each note, can also be list of lists

    Returns
    -------
    onset : int
        Integer corresponding to the index of the first note

    '''
    if isinstance(notes[0], list):
        onsets = []
        for voice in notes:
            for i in range(len(voice)):
                if voice[i] != None:
                    onset.append(i)
        return min(onsets)
    else:
        for i in range(len(notes)):
            if notes[i] != None:
                return i
            
            
        

def transpose(notes, n_semitones, mod = True, ascending = True):
    '''
    Function to transpose notes up or down
    
    Parameters
    ----------
    notes : list of int
        
    n_semitones: int
        Indicates how many note steps to transpose by
        
    mod: bool
        Indicates whether to mod 12 the result (i.e. to return relative position in a scale) or not (i.e. to return absolute position)
        
    ascending: bool
        Indicates whether to transpose up or down

    Returns
    -------
        A list of int representing the transposed notes

    '''
    if ascending: #ascending is a boolean
        diff = int(n_semitones)
    else:
        diff = int(-n_semitones)
    try:
        result = [note + diff for note in notes]
        if mod:
            result = [note % 12 for note in result]
    except TypeError:
        result = notes + diff
        if mod:
            result %= 12
    return result

def extend_range(notes, start_octave = -3, end_octave = 4):
    '''
    Function to obtain an extended list of notes across start_octave to end_octave from a given list
    
    Parameters
    ----------
    notes : list of int
        
    start_octave: int
        Indicates which start octave (w.r.t. the input octave) the function should transpose each note by
        
    end_octave: int
        Indicates which end octave (w.r.t. the input octave) the function should transpose each note by

    Returns
    -------
        A list of int representing the notes, replicated across octave ranging from start_octave to end_octave

    '''
    result = []
    try:
        for note in notes:
            result += [note + 12*i for i in range(start_octave, end_octave)]
    except TypeError:
        result = notes + [12*i for i in range(start_octave, end_octave)]
    return result   
    
def func_get_progression_costs(filename):  #"chord_progression_major_v1.csv"
    '''
    Reads chord progression costs into a dictionary.
    Dictionary key is a tuple of (chord1, chord2) indicating the chord progression from chord1 to chord 2
    Dictionary value is the weight value of chord progressions, which should be between 0.0 and 1.0 (with 0.1 step) inclusive.

    '''
    df = pd.read_csv("../data/"+filename, header=1, index_col=0)
    reset_df = df.stack().reset_index()
    dic = {}
    for index, row in reset_df.iterrows():
        dic[int(row["Chord_1"]),int(row["level_1"])] = row[0]
    return(dic)

def infer_key_tonality(notes, final_note_weight = 1, verbose = False):
    '''
    Function to infer the key and tonality of music given a list of notes. The idea is to count the number of notes that fall within the scale of the key being considered (positive score), and the number of notes that do not fall within the key being considered (penalty score).
    
    Parameters
    ----------
    notes : list of int
        Can be a list representing a melody.
        Can be several lists (e.g. 4 lists representing each voice part) reshaped into a long list. In this case, the last segment of the list should correspond to the melody. Otherwise, final_note_weight should be set to 0.
        
    final_note_weight : positive int/float
        Usually, the last note in a melody indicates the key, but not always, especially if the input notes are only an excerpt and not the entire piece. If excerpt, set final_note_weight = 0. See "notes" for the other case to set value to 0.
        
    verbose : bool
        Indicates whether to print and return the evaluation and penalty score for each key

    Returns
    -------
    key : int from 0 (indicating "C" key) to 11 (indicating "B" key). Each increment of 1 represents the key one semitone higher.
    
    tonality : "major" or "minor"
    
    if verbose:
        key_scores : dict, with dict key being the musical key int, and the value being a list of [major_score, minor_score].
            E.g. {0: [35, 40]} indicates that 35 notes fall in the C major scale and 40 notes fall in the C minor scale (adjusted by final_note_weight).

        key_penalties : dict, with dict key being the musical key int, and the value being a list of [major_penalty, minor penalty].
            E.g. {2: [7, 10]} indicates that 7 notes fall outside the D major scale and 10 notes fall outside the D minor scale (adjusted by final_note_weight).

    '''
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    minor_scale = [0, 2, 3, 5, 7, 8, 9, 10, 11]
    key_scores = {}
    key_penalties = {}
    best_scores = {}
    lowest_penalties = {}
    tonalities = {}
    for n in range(11):
        new_notes = transpose(notes, n, ascending = False)
        major_score = sum(x in major_scale for x in new_notes)
        minor_score = sum(x in minor_scale for x in new_notes)
        major_penalty = sum(x not in major_scale for x in new_notes)
        minor_penalty = sum(x not in minor_scale for x in new_notes)
        if new_notes[-1] == 0:
            major_score += final_note_weight
            minor_score += final_note_weight
            major_penalty -= final_note_weight
            minor_penalty -= final_note_weight
        if major_score > minor_score and major_penalty < minor_penalty:
            tonality = "major"
        elif minor_score > major_score and minor_penalty < major_penalty:
            tonality = "minor"
        else:
            tonality = "ambiguous"
        key_scores[n] = [major_score, minor_score]
        key_penalties[n] = [major_penalty, minor_penalty]
        best_scores[n] = max(major_score, minor_score)
        lowest_penalties[n] = min(major_penalty, minor_penalty)
        tonalities[n] = tonality

    overall_lowest_penalty = min(lowest_penalties.values())
    overall_best_score = max(best_scores.values())
    key = [k1 for (k1, v1), (k2, v2) in zip(best_scores.items(), lowest_penalties.items()) if v1 == overall_best_score and v2 == overall_lowest_penalty]
    tonality = [tonalities[k] for k in key]
    
    if verbose:
        print('Scores for each key: {}'.format(key_scores))
        print('Penalties for each key: {}'.format(key_penalties))
        return key, tonality, key_scores, key_penalties
    else:
        return key, tonality

def encode_constraints(hard_constraints, soft_constraint_weights):
    '''
    Function to succinctly encode which hard constraints were used, which soft constraints were used and their weights. Function to be used for file naming.
    
    Parameters
    ----------
    hard_constraints : list of bool/int or dict of {constraint name: bool/int}, indicating if each hard constraint is implemented or not
        if list, the values must be in the same order as "hard_constraint_options" indicated within this function
        if value is int, positive values indicate the constraint is implemented, non-positive values indicate not implemented
        
    soft_constraint_weights : list of ints or dict of {constraint name: int}, indicating the weights of soft constraint
        if list, the values must be in the same order as "soft_constraint_options" indicated within this function
        A negative float weight indicates that the corresponding soft constraint is not implemented
        Non-negative weights must be between 1 and 99 inclusive

    Returns
    -------
    string1 : string encoding of hard constraints
    
    string2 : string encoding of soft constraint weights

    '''
    #hard_constraints and soft_constraints can be a list of Booleans or Integers, or a dictionary of Booleans or Integers
    hard_constraint_options = ['musical input', 'voice range', 'chord membership', 'first last chords',
                               'chord repetition','chord bass repetition', 'adjacent bar chords', 'voice crossing', 'parallel movement',
                               'chord spacing']
    soft_constraint_options = ['chord progression', 'chord bass repetition', 'leap resolution',
                               'melodic movement', 'note repetition', 'parallel movement', 'voice overlap', 'adjacent bar chords',
                               'chord spacing', 'distinct notes', 'voice crossing', 'voice range']
    bitstring1 = ''
    error = False
    if isinstance(hard_constraints, list):
        for c in hard_constraints:
            if c > 0:
                bitstring1 += '1'
            else:
                bitstring1 += '0'
    elif isinstance(hard_constraints, dict):
        for c in hard_constraint_options:
            if c in hard_constraints:
                if hard_constraints[c] > 0:
                    bitstring1 += '1'
                else:
                    bitstring1 += '0'
            else:
                bitstring1 += '0'
    else:
        error = True
        print('Error: Unrecognised argument data type for hard constraints')

    string2 = ''
    if isinstance(soft_constraint_weights, list):
        for w in soft_constraint_weights:
            if w > 0:
                string2 += str(int(w))
            elif w >= 1:
                print('Error: Soft constraint weights must be between 1 and 100')
            else:
                string2 += '00'
    elif isinstance(soft_constraint_weights, dict):
        for w in soft_constraint_options:
            if w in soft_constraint_weights:
                if soft_constraint_weights[w] > 0:
                    string2 += str(int(soft_constraint_weights[w]))
                elif soft_constraint_weights[w] >= 1:
                    print('Error: Soft constraint weights must be between 1 and 100')
                else:
                    string2 += '00'
            else:
                string2 += '00'
    else:
        error = True
        print('Error: Unrecognised argument data type for soft constraints')
    
    if error:
        return None
    else:
        string1 = str(int(bitstring1, 2))
        return string1, string2

def decode_constraints(hard_constraint_string, soft_constraint_string, data_type = 'list'):
    '''
    Function decode an encoded integer into which constraints were implemented, and the soft constraint weights.
    
    Parameters
    ----------
    hard_constraint_string : a string as encoded by the "encode_constraints" function
    
    soft_constraint_string : a string as encoded by the "encode_constraints" function
        
    data_type : 'list' or 'dict'
        Indicates whether the function should return results as lists or dictionaries.

    Returns
    -------
        hard_constraints : Indicates whether a hard constraint is implemented or not
            if data_type is 'list', a list of bools in the order of "hard_constraint_options" in this function
            if data_type is 'dict', a dict of bools of the form {hard constraint name: bool}
        
        soft_constraint_weights : Indicates the weights as int (from 1 to 99 inclusive) of each soft_constraint
            if data_type is 'list', a list of weights in the order of "soft_constraint_options" in this function
            if data_type is 'dict', a dict of weights of the form {soft constraint name: weight}

    '''
    hard_constraint_options = ['musical input', 'voice range', 'chord membership', 'first last chords',
                               'chord repetition','chord bass repetition', 'adjacent bar chords', 'voice crossing', 'parallel movement',
                               'chord spacing']
    soft_constraint_options = ['chord progression', 'chord bass repetition', 'leap resolution',
                               'melodic movement', 'note repetition', 'parallel movement', 'voice overlap', 'adjacent bar chords',
                               'chord spacing', 'distinct notes', 'voice crossing', 'voice range']
    string1 = bin(int(hard_constraint_string))
                              
    flag = 0
    for s in string1:
        flag += 1
        if s == 'b':
            break
    string1 = string1[flag:]
    
    if data_type == 'list':
        hard_constraints, soft_constraint_weights = [], []
    elif data_type == 'dict':
        hard_constraints, soft_constraint_weights = {}, {}
    else:
        print('Error: Unrecognised argument provided. data_type must be \'list\' or \'dict\'')
        return None
    
    for i, s in enumerate(string1):
        #print(i, 'hard', bool(int(s)))
        if data_type == 'list':
            hard_constraints.append(bool(s))
        else:
            hard_constraints[hard_constraint_options[i]] = bool(s)

    for i, c in enumerate(soft_constraint_options):
        w = int(soft_constraint_string[i:2*(i+1)])
        if w == 0:
            w = -1
        if data_type == 'list':
            soft_constraint_weights.append(w)
        else:
            soft_constraint_weights[c] = w
    return hard_constraints, soft_constraint_weights

def func_get_best_progression_chord(filename, direction="fwd"):  #"chord_progression_major_v1.csv"

    df=pd.read_csv("../data/"+filename, header=1, index_col=0)
    N_chords=len(df)
    reset_df=df.stack().reset_index()
    best_chord={}
    if direction=="fwd":
        for i in range(N_chords):
            min_cost=1
            
            for index,row in reset_df.iterrows(): 
                if int(row["Chord_1"])==i and int(row["level_1"])!= int(row["Chord_1"]) and row[0]< min_cost:
                    min_cost=row[0]
                    best_chord[i]= int(row["level_1"])
    else: #backward
        for i in range(N_chords):
            min_cost=1
            
            for index,row in reset_df.iterrows(): 
                if int(row["level_1"])==i and int(row["level_1"])!= int(row["Chord_1"])  and row[0]< min_cost:
                    min_cost=row[0]
                    best_chord[i]= int(row["Chord_1"])           
    return best_chord

if __name__ == '__main__':
    print(func_get_best_progression_chord("chord_progression_major_v1.csv" ))
    print(func_get_best_progression_chord("chord_progression_major_v1.csv" ,"bwd"))
            