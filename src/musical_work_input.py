import src.music_functions

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
    
    def transpose_work(self, n_semitones, ascending):
        self.melody = src.music_functions.transpose(self.melody, n_semitones, mod = False, ascending = ascending)
        self.key = src.music_functions.transpose(self.key, n_semitones, mod = True, ascending = ascneding)

    def update_reference_note(self, new_reference_note):
        diff = new_reference_note - self.reference_note
        self.transpose_work(diff, ascending = True)
        self.reference_note = new_reference_note