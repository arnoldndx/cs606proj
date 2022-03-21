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