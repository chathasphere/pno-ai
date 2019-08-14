import os
import pdb
from pretty_midi import PrettyMIDI, Instrument
from preprocess import PreprocessingPipeline

def write_to_midi(note_sequences, output_dir):
    i = 1
    for note_sequence in note_sequences:
        midi = PrettyMIDI(initial_tempo = 80)
        piano = Instrument(program=0, is_drum=False, name="test{}".format(i))
        piano.notes = note_sequence
        midi.instruments.append(piano)
        output_name = output_dir + "/test{}.midi".format(i)
        i += 1
        #with open(output_name, 'w')
        midi.write(output_name)
    print("Piano data successfully extracted from midis, navigate to {} to listen"\
            .format(output_dir))

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
    pipeline = PreprocessingPipeline(input_dir = "data/test", split_size = 30)
    pipeline.run()
    # write_to_midi(pipeline.note_sequences, "output/test")
    check_sample_lengths(pipeline.split_samples, 30)
    # pdb.set_trace()


if __name__ == "__main__":
    main()
