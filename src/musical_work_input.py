class MusicalWorkInput:
    def __init__(self, title, meter, key, tonality, first_on_beat, melody, reference_note = 24):
        self.title = title #string
        self.meter = meter #integer
        self.key = key #integer, where 0 refers to C, 1 refers to C#/Db, ..., 11 refers to B
        self.tonality = tonality #'major' or 'minor'
        self.first_on_beat = first_on_beat #integer not exceeding meter
        self.melody = melody #list of integers
        self.reference_note = reference_note #refers to which integer corresponds to middle C, i.e. C4
        self.melody_len = len(self.melody)
    
    def update_reference_note(self, new_reference_note):
        diff = new_reference_note - self.reference_note
        self.melody = [note + diff for note in self.melody]
        self.key = (self.key + diff) % 12
        self.reference_note = new_reference_note
    
    def transpose(self, n_semitones, ascending):
        if ascending: #ascending is a boolean
            diff = n_semitones
        else:
            diff = -n_semitones
        self.melody = [note + diff for note in self.melody]
        self.key = (self.key + diff) % 12

