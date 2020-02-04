import argparse, uuid, subprocess
import torch
from model import MusicTransformer
from preprocess import SequenceEncoder
from helpers import sample, write_midi
import midi_input
import yaml

class GeneratorError(Exception):
    pass

def main():
    parser = argparse.ArgumentParser("Script to generate MIDI tracks by sampling from a trained model.")

    parser.add_argument("--model_key", type=str, 
            help="Key in saved_models/model.yaml, helps look up model arguments and path to saved checkpoint.")
    parser.add_argument("--sample_length", type=int, default=512,
            help="number of events to generate")
    parser.add_argument("--temps", nargs="+", type=float, 
            default=[1.0],
            help="space-separated list of temperatures to use when sampling")
    parser.add_argument("--n_trials", type=int, default=3,
            help="number of MIDI samples to generate per experiment")
    parser.add_argument("--live_input", action='store_true', default = False,
            help="if true, take in a seed from a MIDI input controller")

    parser.add_argument("--play_live", action='store_true', default=False,
            help="play sample(s) at end of script if true")
    parser.add_argument("--keep_ghosts", action='store_true', default=False)
    parser.add_argument("--stuck_note_duration", type=int, default=0)

    args=parser.parse_args()

    model_key = args.model_key

    try:
        model_dict = yaml.safe_load(open('saved_models/model.yaml'))[model_key]
    except:
        raise GeneratorError(f"could not find yaml information for key {model_key}")

    model_path = model_dict["path"]
    model_args = model_dict["args"]
    try:
        state = torch.load(model_path)
    except RuntimeError:
        state = torch.load(model_path, map_location="cpu")
    
    n_velocity_events = 32
    n_time_shift_events = 125

    decoder = SequenceEncoder(n_time_shift_events, n_velocity_events,
           min_events=0)

    if args.live_input:
        print("Expecting a midi input...")
        note_sequence = midi_input.read(n_velocity_events, n_time_shift_events)
        prime_sequence = decoder.encode_sequences([note_sequence])[0]

    else:
        prime_sequence = []

    model = MusicTransformer(**model_args)
    model.load_state_dict(state, strict=False)

    temps = args.temps

    trial_key = str(uuid.uuid4())[:6]
    n_trials = args.n_trials

    keep_ghosts = args.keep_ghosts
    stuck_note_duration = None if args.stuck_note_duration == 0 else args.stuck_note_duration

    for temp in temps:
        print(f"sampling temp={temp}")
        note_sequence = []
        for i in range(n_trials):
            print("generating sequence")
            output_sequence = sample(model, prime_sequence = prime_sequence, sample_length=args.sample_length, temperature=temp)
            note_sequence = decoder.decode_sequence(output_sequence, 
                verbose=True, stuck_note_duration=None)

            output_dir = f"output/{model_key}/{trial_key}/"
            file_name = f"sample{i+1}_{temp}"
            write_midi(note_sequence, output_dir, file_name)

    for temp in temps:      
        try:
            subprocess.run(['timidity', f"output/{model_key}/{trial_key}/sample{i+1}_{temp}.midi"])
        except KeyboardInterrupt:
            continue

if __name__ == "__main__":
    main()
