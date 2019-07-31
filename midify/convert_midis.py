#!/usr/bin/env python3
import argparse, os
import pdb
from pretty_midi import PrettyMIDI
import six

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise argparse.ArgumentTypeError("'{}' not a valid directory".format(string))


def parse_args(args):
    parser = argparse.ArgumentParser(description = "Parse MIDI files in a directory")
    parser.add_argument("-i","--input_dir", type = dir_path, action="store", dest = "input_dir", help="Directory containing midi files to convert")
    parser.add_argument("-o", "--output_dir", type = str, action="store", dest = "output_dir", help="Name of folder in which to store parsed outputs")
    return parser.parse_args()

def convert_files():
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
                pretty_midis.append(PrettyMIDI(midi_str))
            except:
                print("Could not parse {}".format(m))
    return pretty_midis

def main(args):
    parsed = parse_args(args)
    input_dir = parsed.input_dir
    os.chdir(input_dir)
    midis = convert_files()
    print(len(midis))
    #https://github.com/tensorflow/magenta/blob/master/magenta/music/midi_io.py
    #check out line 85 and on to see what data to extract



if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
