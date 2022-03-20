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