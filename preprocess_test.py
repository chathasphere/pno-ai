import os
import pdb
from pretty_midi import PrettyMIDI, Instrument
from preprocess import PreprocessingPipeline
import pathlib

def write_to_midi(note_sequences, output_dir, n_to_write=None):

    if len(note_sequences) == 0:
        print("No note sequences to write out...")
        return
    #number of sequences to write out as MIDI files
    if n_to_write is None:
        n_to_write = len(note_sequences)
    #make the output directory if it doesn't already exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
    for i in range(n_to_write):
    #for note_sequence in note_sequences:
        midi = PrettyMIDI(initial_tempo = 80)
        piano = Instrument(program=0, is_drum=False, name="test{}".format(i))
        piano.notes = note_sequences[i]
        midi.instruments.append(piano)
        output_name = output_dir + "/test{}.midi".format(i)
        #with open(output_name, 'w')
        midi.write(output_name)
    print("Piano data successfully extracted from midis, navigate to {} to listen"\
            .format(output_dir))

def check_ordering(note_sequences):
    for note_sequence in note_sequences:
        ordered = True
        current_time = 0
        for note in note_sequence:
            if note.start < current_time:
                ordered = False
            current_time = note.start
        try: 
            assert ordered
        except AssertionError:
            pdb.set_trace()

def check_sample_lengths(split_samples, split_size):
    for i in range(len(split_samples)):
        sample = split_samples[i]
        sample_end = sample[-1].end
        sample_start = sample[0].start
        try:
            assert sample_end - sample_start <= split_size
        except AssertionError:
            print("Error with sample {}; start is {.2f} and end is {.2f}"\
                    .format(i, sample_end, sample_start))
    print("All samples are less than {} seconds in length.".format(split_size))

def main():
    pipeline = PreprocessingPipeline(input_dir = "data/quick_test", split_size = 30,
            n_velocity_bins = 32)
    pipeline.run()
    check_ordering(pipeline.note_sequences)
    print("Note sequences in order")
    check_ordering(pipeline.split_samples)
    print("Split samples in order")
    #write_to_midi(pipeline.note_sequences, "output/test_midis")
    # check_sample_lengths(pipeline.split_samples, 30)
    #write_to_midi(pipeline.split_samples, "output/test_samples", n_to_write=20)
    encoded_sequences = pipeline.encoded_sequences

if __name__ == "__main__":
    main()
