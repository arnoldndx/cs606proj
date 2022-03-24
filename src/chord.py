class Chord:
    
    def __init__(self, index, name, note_intervals):
        self.name = name
        self.note_intervals = note_intervals #list of integers
        self.size = len(note_intervals)
        self.index = index
        

