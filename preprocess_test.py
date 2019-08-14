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

def main():
    pipeline = PreprocessingPipeline(input_dir = "data/test")
    pipeline.run()
    write_to_midi(pipeline.note_sequences, "output/test")

if __name__ == "__main__":
    main()
