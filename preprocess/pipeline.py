import os, random, copy
import pretty_midi
from pretty_midi import ControlChange
import six
from .sequence_encoder import SequenceEncoder
import numpy as np
from helpers import vectorize

class PreprocessingError(Exception):
    pass

class PreprocessingPipeline():
    #set a random seed
    SEED = 1811
    """
    Pipeline to convert MIDI files to cleaned Piano Midi Note Sequences, split into 
    a more manageable length.
    Applies any sustain pedal activity to extend note lengths. Optionally augments
    the data by transposing pitch and/or stretching sample speed. Optionally quantizes
    timing and/or dynamics into smaller bins.

    Attributes:
        self.split_samples (dict of lists): when the pipeline is run, has two keys, "training" and "validation," each holding a list of split MIDI note sequences.
        self.encoded_sequences (dict of lists): Keys are "training" and "validation." Each holds a list of encoded event sequences, a sparse numeric representation of a MIDI sample.
    """
    def __init__(self, input_dir, stretch_factors = [0.95, 0.975, 1, 1.025, 1.05],
            split_size = 30, sampling_rate = 125, n_velocity_bins = 32,
            transpositions = range(-3,4), training_val_split = 0.9, 
            max_encoded_length = 512, min_encoded_length = 33):
        self.input_dir = input_dir
        self.split_samples = dict()
        self.stretch_factors = stretch_factors
        #size (in seconds) in which to split midi samples
        self.split_size = split_size
        #In hertz (beats per second), quantize sample timings to this discrete frequency
        #So a sampling rate of 125 hz means a smallest time steps of 8 ms
        self.sampling_rate = sampling_rate
        #Quantize sample dynamics (Velocity 1-127) to a smaller number of bins
        #this should be an *integer* dividing 128 cleanly: 2,4,8,16,32,64, or 128. 
        self.n_velocity_bins = n_velocity_bins
        self.transpositions = transpositions
        
        #Fraction of raw MIDI data that goes to the training set
        #the remainder goes to validat
        self.training_val_split = training_val_split

        self.encoder = SequenceEncoder(n_time_shift_events = sampling_rate,
                n_velocity_events = n_velocity_bins, 
                min_events = min_encoded_length,
                max_events = max_encoded_length)
        self.encoded_sequences = dict()

        random.seed(PreprocessingPipeline.SEED)

        """
        Args:
            input_dir (str): path to input directory. All .midi or .mid files in this directory will get processed.
            stretch_factors (list of float): List of constants by which note end times and start times will be multiplied. A way to augment data.
            split_size (int): Max length, in seconds, of samples into which longer MIDI note sequences are split.
            sampling_rate (int): How many subdivisions of 1,000 milliseconds to quantize note timings into. E.g. a sampling rate of 100 will mean end and start times are rounded to the nearest 0.01 second.
            n_velocity_bins (int): Quantize 128 Midi velocities (amplitudes) into this many bins: e.g. 32 velocity bins mean note velocities are rounded to the nearest multiple of 4.
            transpositions (iterator of ints): Transpose note pitches up/down by intervals (number of half steps) in this iterator. Augments a dataset with transposed copies.
            training_val_split (float): Number between 0 and 1 defining the proportion of raw data going to the training set. The rest goes to validation.
            max_encoded_length (int): Truncate encoded samples containing more
            events than this number.
            min_encoded_length (int): Discard encoded samples containing fewer events than this number.
        """


    def run(self):
        """
        Main pipeline call...parse midis, split into test and validation sets,
        augment, quantize, sample, and encode as event sequences. 
        """
        midis = self.parse_files(chdir=True) 
        total_time = sum([m.get_end_time() for m in midis])
        print("\n{} midis read, or {:.1f} minutes of music"\
                .format(len(midis), total_time/60))

        note_sequences = self.get_note_sequences(midis)
        del midis
        #vectorize note sequences
        note_sequences = [vectorize(ns) for ns in note_sequences]
        print("{} note sequences extracted\n".format(len(note_sequences)))
        self.note_sequences = self.partition(note_sequences)
        for mode, sequences in self.note_sequences.items():
            print(f"Processing {mode} data...")
            print(f"{len(sequences):,} note sequences")
            if mode == "training":
                sequences = self.stretch_note_sequences(sequences)
                print(f"{len(sequences):,} stretched note sequences")
            samples = self.split_sequences(sequences)
            self.quantize(samples)
            print(f"{len(samples):,} quantized, split samples")
            if mode == "training":
                samples = self.transpose_samples(samples)
                print(f"{len(samples):,} transposed samples")
            self.split_samples[mode] = samples
            self.encoded_sequences[mode] = self.encoder.encode_sequences(samples)
            print(f"Encoded {mode} sequences!\n")

    def parse_files(self, chdir=False):
        """
        Recursively parse all MIDI files in a given directory to 
        PrettyMidi objects.
        """
        if chdir: 
            home_dir = os.getcwd()
            os.chdir(self.input_dir)

        pretty_midis = []
        folders = [d for d in os.listdir(os.getcwd()) if os.path.isdir(d)]
        if len(folders) > 0:
            for d in folders:
                os.chdir(d)
                pretty_midis += self.parse_files()
                os.chdir("..")
        midis = [f for f in os.listdir(os.getcwd()) if \
                (f.endswith(".mid") or f.endswith("midi"))]
        print(f"Parsing {len(midis)} midi files in {os.getcwd()}...")
        for m in midis:
            with open(m, "rb") as f:
                try:
                    midi_str = six.BytesIO(f.read())
                    pretty_midis.append(pretty_midi.PrettyMIDI(midi_str))
                    #print("Successfully parsed {}".format(m))
                except:
                    print("Could not parse {}".format(m))
        if chdir:
            os.chdir(home_dir)

        return pretty_midis

    def get_note_sequences(self, midis):
        """
        Given a list of PrettyMidi objects, extract the Piano track as a list of 
        Note objects. Calls the "apply_sustain" method to extract the sustain pedal
        control changes.
        """

        note_sequences = []
        for m in midis:
            if m.instruments[0].program == 0:
                piano_data = m.instruments[0]
            else:
                #todo: write logic to safely catch if there are non piano instruments,
                #or extract the piano midi if it exists
                raise PreprocessingError("Non-piano midi detected")
            note_sequence = self.apply_sustain(piano_data)
            note_sequence = sorted(note_sequence, key = lambda x: (x.start, x.pitch))
            note_sequences.append(note_sequence)

        return note_sequences



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
        first_sustain_control = next((c for c in control_changes if c.number == 64),
                ControlChange(number=64, value=0, time=0))

        if first_sustain_control.value >= 64:
            sustain_position = _SUSTAIN_ON
        else:
            sustain_position = _SUSTAIN_OFF
        #if for some reason pedal was not touched...
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
                    try:
                        live_notes.remove(note)
                    except ValueError:
                        print("***Unexpected note sequence...possible duplicate?")
                        pass
        return notes

    def partition(self, sequences):
       """
       Partition a list of Note sequences into a training set and validation set.
       Returns a dictionary {"training": training_data, "validation": validation_data}
       """
       partitioned_sequences = {}
       random.shuffle(sequences)

       n_training = int(len(sequences) * self.training_val_split)
       partitioned_sequences['training'] = sequences[:n_training]
       partitioned_sequences['validation'] = sequences[n_training:]

       return partitioned_sequences

    def stretch_note_sequences(self, note_sequences):
        """
        Stretches tempo (note start and end time) for each sequence in a given list
        by each of the pipeline's stretch factors. Returns a list of Note sequences.
        """
        stretched_note_sequences = []
        for note_sequence in note_sequences:
            for factor in self.stretch_factors:
                if factor == 1:
                    stretched_note_sequences.append(note_sequence)
                    continue
                stretched_sequence = np.copy(note_sequence)
                #stretch note start time
                stretched_sequence[:,0] *= factor
                #stretch note end time
                stretched_sequence[:,1] *= factor
                stretched_note_sequences.append(stretched_sequence)

        return stretched_note_sequences


    def split_sequences(self, sequences):
        """
        Given a list of Note sequences, splits them into samples no longer than 
        a given length. Returns a list of split samples.
        """

        samples = []
        if len(sequences) == 0:
            raise PreprocessingError("No note sequences available to split")

        for note_sequence in sequences:
            sample_length = 0
            sample = []
            i = 0
            while i < len(note_sequence):
                note = np.copy(note_sequence[i])
                if sample_length == 0:
                    sample_start = note[0]
                    if note[1] > self.split_size + sample_start:
                        #prevent case of a zero-length sample
                        #print(f"***Current note has length of more than {self.split_size} seconds...reducing duration")
                        note[1] = sample_start + self.split_size
                    sample.append(note)
                    sample_length = self.split_size
                else:
                    if note[1] <= sample_start + self.split_size:
                        sample.append(note)
                        if note[1] > sample_start + sample_length:
                            sample_length = note[1] - sample_start
                    else:
                        samples.append(np.asarray(sample))
                        #sample start should begin with the beginning of the
                        #*next* note, how do I handle this...
                        sample_length = 0
                        sample = []
                i += 1
        return samples

    def quantize(self, samples):
        """
        Quantize timing and dynamics in a Note sample in place. This converts continuous
        time to a discrete, encodable quantity and simplifies input for the model.
        Quantizes note start/ends to a smallest perceptible timestep (~8ms) and note
        velocities to a few audibly distinct bins (around 32).
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
        for sample in samples:
            sample_start_time = next((note[0] for note in sample), 0)
            for note in sample:
                #reshift note start and end times to begin at zero
                note[0] -= sample_start_time
                note[1] -= sample_start_time
                #delete this 
                if note[0] < 0 or note[1] < 0:
                    raise PreprocessingError
                if timestep:
                    #quantize timing
                    note[0] = (note[0] * self.sampling_rate) // 1 * timestep
                    note[1] = (note[1] * self.sampling_rate) // 1 * timestep
                if velocity_step:
                    #quantize dynamics
                    #smallest velocity is 1 (otherwise we can't hear it!)
                    note[3] = (note[3] // velocity_step *\
                            velocity_step) + 1

    def transpose_samples(self, samples):
        """
        Transposes the pitch of a sample note by note according to a list of intervals.
        """
        transposed_samples = []
        for sample in samples:
            for transposition in self.transpositions:
                if transposition == 0:
                    transposed_samples.append(sample)
                    continue
                transposed_sample = np.copy(sample)
                #shift pitches in sample by transposition
                transposed_sample[:,2] += transposition
                #should I adjust pitches that fall out of the range of 
                #a piano's 88 keys? going to be pretty uncommon.
                transposed_samples.append(transposed_sample)

        return transposed_samples



