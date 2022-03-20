import pandas as pd

def transpose(notes, n_semitones, mod = True, ascending = True):
    if ascending: #ascending is a boolean
        diff = n_semitones
    else:
        diff = -n_semitones
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
    result = []
    try:
        for note in notes:
            result += [note + 12*i for i in range(start_octave, end_octave)]
    except TypeError:
        result = notes + [12*i for i in range(start_octave, end_octave)]
    return result   
    
def func_get_progression_costs(filename):  #"chord_progression_major_v1.csv"

    df=pd.read_csv("../data/"+filename, header=1, index_col=0)
    reset_df=df.stack().reset_index()
    dic={}
    for index,row in reset_df.iterrows():
        
        dic[int(row["Chord_1"]),int(row["level_1"])]=row[0]
    return(dic)

def infer_key_tonality(notes, final_note_weight = 1, verbose = False):
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

def encode_constraints(hard_constraints, soft_constraints):
    #hard_constraints and soft_constraints can be a list of Booleans or Integers, or a dictionary of Booleans or Integers
    hard_constraint_options = ['musical input', 'voice range', 'chord membership', 'first last chords',
                               'chord repetition','chord bass repetition', 'adjacent bar chords', 'voice crossing', 'parallel movement',
                               'chord spacing']
    soft_constraint_options = ['chord progression', 'chord bass repetition', 'leap resolution',
                               'melodic movement', 'note repetition', 'parallel movement', 'voice overlap', 'adjacent bar chords',
                               'chord spacing', 'distinct notes', 'voice crossing', 'voice range']
    bitstring = ''
    error = False
    if isinstance(hard_constraints, list):
        for c in hard_constraints:
            if c > 0:
                bitstring += '1'
            else:
                bitstring += '0'
    elif isinstance(hard_constraints, dict):
        for c in hard_constraint_options:
            if c in hard_constraints:
                if hard_constraints[c] > 0:
                    bitstring += '1'
                else:
                    bitstring += '0'
            else:
                bitstring += '0'
    else:
        error = True
        print('Error: Unrecognised argument data type for hard constraints')
        
    if isinstance(soft_constraints, list):
        for c in soft_constraints:
            if c > 0:
                bitstring += '1'
            else:
                bitstring += '0'
    elif isinstance(soft_constraints, dict):
        for c in soft_constraint_options:
            if c in soft_constraints:
                if soft_constraints[c] > 0:
                    bitstring += '1'
                else:
                    bitstring += '0'
            else:
                bitstring += '0'
    else:
        error = True
        print('Error: Unrecognised argument data type for soft constraints')
    
    if error:
        return None
    else:
        return int(bitstring, 2)

def decode_constraints(integer, data_type = 'list'):
    hard_constraint_options = ['musical input', 'voice range', 'chord membership', 'first last chords',
                               'chord repetition','chord bass repetition', 'adjacent bar chords', 'voice crossing', 'parallel movement',
                               'chord spacing']
    soft_constraint_options = ['chord progression', 'chord bass repetition', 'leap resolution',
                               'melodic movement', 'note repetition', 'parallel movement', 'voice overlap', 'adjacent bar chords',
                               'chord spacing', 'distinct notes', 'voice crossing', 'voice range']
    bitstring = bin(integer)
    flag = 0
    for s in bitstring:
        flag += 1
        if s == 'b':
            break
    bitstring = bitstring[flag:]
    
    if data_type == 'list':
        hard_constraints, soft_constraints = [], []
    elif data_type == 'dict':
        hard_constraints, soft_constraints = {}, {}
    else:
        print('Error: Unrecognised argument provided. data_type must be \'list\' or \'dict\'')
        return None
    
    for i, s in enumerate(bitstring):
        if i < len(hard_constraint_options):
            print(i, 'hard', bool(int(s)))
            if data_type == 'list':
                hard_constraints.append(bool(int(s)))
            else:
                hard_constraints[hard_constraint_options[i]] = bool(int(s))
        else:
            print(i, 'soft', bool(int(s)))
            if data_type == 'list':
                soft_constraints.append(bool(int(s)))
            else:
                soft_constraints[soft_constraint_options[i]] = bool(int(s))
    return hard_constraints, soft_constraints
            