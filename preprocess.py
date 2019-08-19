import os
import pretty_midi
import six
import copy
from sequence_encoder import SequenceEncoder

class PreprocessingError(Exception):
    pass

class PreprocessingPipeline():
    """
    Pipeline to convert MIDI files to cleaned Piano Midi Note Sequences, split into 
    a more manageable length.
    Applies any sustain pedal activity to extend note lengths. Optionally augments
    the data by transposing pitch and/or stretching sample speed. Optionally quantizes
    timing and/or dynamics into smaller bins.
    """
    def __init__(self, input_dir, stretch_factors = [0.95, 0.975, 1, 1.025, 1.05],
            split_size = 30, sampling_rate = 125, n_velocity_bins = 32,
            transpositions = range(-3,4)):
        self.input_dir = input_dir
        self.note_sequences = []
        #size (in seconds) in which to split midi samples
        self.split_samples = []
        self.stretch_factors = stretch_factors
        self.split_size = split_size
        self.quantized_samples = []
        #In hertz (beats per second), quantize sample timings to this discrete frequency
        #So a sampling rate of 125 hz means a smallest time steps of 8 ms
        self.sampling_rate = sampling_rate
        #Quantize sample dynamics (Velocity 1-127) to a smaller number of bins
        #this should be an *integer* dividing 128 cleanly: 2,4,8,16,32,64, or 128. 
        self.n_velocity_bins = n_velocity_bins
        self.transpositions = transpositions

        self.encoder = SequenceEncoder(n_time_shift_events = sampling_rate,
                n_velocity_events = n_velocity_bins)
        self.encoded_sequences = []


    def run(self):
        home_dir = os.getcwd()
        os.chdir(self.input_dir)
        midis = self.convert_files() 
        os.chdir(home_dir)
        total_time = sum([m.get_end_time() for m in midis])
        print("{} midis read, or {:.1f} minutes of music"\
                .format(len(midis), total_time))

        self.get_note_sequences(midis)
        print("{} note sequences ".format(len(self.note_sequences)))
        self.stretch_note_sequences()
        print("{} time-stretched note sequences".format(len(self.note_sequences)))
        self.split_sequences()
        self.quantize_samples()
        print("{} quantized, split samples".format(len(self.split_samples)))
        self.transpose_samples()
        print("{} transposed samples".format(len(self.split_samples)))
        samples = self.split_samples
        self.encoded_sequences = self.encoder.encode_event_sequences(samples)

    def convert_files(self):
        pretty_midis = []
        folders = [d for d in os.listdir(os.getcwd()) if os.path.isdir(d)]
        if len(folders) > 0:
            for d in folders:
                os.chdir(d)
                pretty_midis += convert_files()
                os.chdir("..")
        midis = [f for f in os.listdir(os.getcwd()) if \
                (f.endswith(".mid") or f.endswith("midi"))]
        for m in midis:
            with open(m, "rb") as f:
                try:
                    midi_str = six.BytesIO(f.read())
                    pretty_midis.append(pretty_midi.PrettyMIDI(midi_str))
                    print("Successfully parsed {}!".format(m))
                except:
                    print("Could not parse {}".format(m))
        return pretty_midis

    def get_note_sequences(self, midis):
        for m in midis:
            if m.instruments[0].program == 0:
                piano_data = m.instruments[0]
            else:
                #todo: write logic to safely catch if there are non piano instruments,
                #or extract the piano midi if it exists
                raise PreprocessingError("Non-piano midi detected")
            self.note_sequences.append(self.apply_sustain(piano_data))


    def apply_sustain(self, piano_data):
        """
        While the sustain pedal is applied during a midi, extend the length of all 
        notes to the beginning of the next note of the same pitch or to 
        the end of the sustain. Returns a midi notes sequence.
        """
        _SUSTAIN_ON = 0
        _SUSTAIN_OFF = 1
        _NOTE_ON = 2
        _NOTE_OFF = 3
 
        notes = copy.deepcopy(piano_data.notes)
        control_changes = piano_data.control_changes
        #sequence of SUSTAIN_ON, SUSTAIN_OFF, NOTE_ON, and NOTE_OFF actions
        first_sustain_control = next(c for c in control_changes if c.number == 64)
        if first_sustain_control.value >= 64:
            sustain_position = _SUSTAIN_ON
        else:
            sustain_position = _SUSTAIN_OFF
        action_sequence = [(first_sustain_control.time, sustain_position, None)]
        #delete this please
        cleaned_controls = []
        for c in control_changes:
            #Ignoring the sostenuto and damper pedals due to complications
            if sustain_position == _SUSTAIN_ON:
                if c.value >= 64:
                    #another SUSTAIN_ON
                    continue
                else:
                    sustain_position = _SUSTAIN_OFF
            else:
                #look for the next on signal
                if c.value < 64:
                    #another SUSTAIN_OFF
                    continue
                else:
                    sustain_position = _SUSTAIN_ON
            action_sequence.append((c.time, sustain_position, None))
            cleaned_controls.append((c.time, sustain_position))
    
        action_sequence.extend([(note.start, _NOTE_ON, note) for note in notes])
        action_sequence.extend([(note.end, _NOTE_OFF, note) for note in notes])
        #sort actions by time and type
    
        action_sequence = sorted(action_sequence, key = lambda x: (x[0], x[1]))
        live_notes = []
        sustain = False
        for action in action_sequence:
            if action[1] == _SUSTAIN_ON:
                sustain = True
            elif action[1] == _SUSTAIN_OFF:
                #find when the sustain pedal is released
                off_time = action[0]
                for note in live_notes:
                    if note.end < off_time:
                        #shift the end of the note to when the pedal is released
                        note.end = off_time
                        live_notes.remove(note)
                sustain = False
            elif action[1] == _NOTE_ON:
                current_note = action[2]
                if sustain:
                    for note in live_notes:
                        # if there are live notes of the same pitch being held, kill 'em
                        if current_note.pitch == note.pitch:
                            note.end = current_note.start
                            live_notes.remove(note)
                live_notes.append(current_note)
            else:
                if sustain == True:
                    continue
                else:
                    note = action[2]
                    live_notes.remove(note)
        return notes

    def stretch_note_sequences(self):
        stretched_note_sequences = []
        for note_sequence in self.note_sequences:
            for factor in self.stretch_factors:
                if factor == 1:
                    stretched_note_sequences.append(note_sequence)
                    continue
                stretched_sequence = copy.deepcopy(note_sequence)
                for note in stretched_sequence:
                    note.start *= factor
                    note.end *= factor
                stretched_note_sequences.append(stretched_sequence)

        self.note_sequences = stretched_note_sequences


    def split_sequences(self):
        if len(self.note_sequences) == 0:
            raise PreprocessingError("No note sequences available to split")

        for note_sequence in self.note_sequences:
            sample_length = 0
            sample = []
            i = 0
            while i < len(note_sequence):
                note = note_sequence[i]
                if sample_length == 0:
                    sample_start = note.start
                new_length = note.end - sample_start
                if new_length <= self.split_size:
                    sample.append(note)
                    sample_length = new_length
                else:
                    if len(sample) == 0:
                        raise PreprocessingError("Sample could not be split!")
                    self.split_samples.append(sample)
                    #sample start should begin with the beginning of the
                    #*next* note, how do I handle this...
                    sample_length = 0
                    sample = []
                i += 1

    def quantize_samples(self):
        """
        Quantize note sequence timing (note start and end) to a smallest timestep,
        typically 8-10 ms. Quantize note sequence dynamics (note velocity) to a smaller
        number of bins, on the order of 32. This step ends up freeing up a lot of memory
        with little audible loss in fidelity.
        """
        #define smallest timestep (in seconds)
        try:
            timestep = 1 / self.sampling_rate
        except ZeroDivisionError:
            timestep = 0
        #define smallest dynamics increment
        try:
            velocity_step = 128 // self.n_velocity_bins
        except ZeroDivisionError:
            velocity_step = 0
        for sample in self.split_samples:
            for note in sample:
                if timestep:
                    #quantize timing
                    note.start = (note.start // timestep) * timestep
                    note.end = (note.end // timestep) * timestep
                if velocity_step:
                    #quantize dynamics
                    #smallest velocity is 1 (otherwise we can't hear it!)
                    note.velocity = (note.velocity // velocity_step *\
                            velocity_step) + 1

    def transpose_samples(self):
        transposed_samples = []
        for sample in self.split_samples:
            for transposition in self.transpositions:
                if transposition == 0:
                    transposed_samples.append(sample)
                    continue
                transposed_sample = copy.deepcopy(sample)
                for note in transposed_sample:
                    new_pitch = note.pitch + transposition
                    if new_pitch > 127:
                        new_pitch -= 12
                    elif new_pitch < 0:
                        new_pitch += 12
                    note.pitch = new_pitch
                transposed_samples.append(transposed_sample)

        self.split_samples = transposed_samples



