class Chord:
    chord_vocab_size = 0
    
    def __init__(self, name, note_intervals):
        self.name = name
        self.note_intervals = note_intervals #set of integers
        self.size = len(note_intervals)
        self.index = Chord.chord_vocab_size
        Chord.chord_vocab_size += 1
        

