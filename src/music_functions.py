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

def extend_range(notes, start_octave = -1, end_octave = 6):
    result = []
    try:
        for note in notes:
            result += [note + 12*i for i in range(start_octave, end_octave)]
    except TypeError:
        result = notes + [12*i for i in range(start_octave, end_octave)]
    return result   
    