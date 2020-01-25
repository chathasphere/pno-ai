import argparse, pathlib, uuid, subprocess
from model import MusicTransformer
from sequence_encoder import SequenceEncoder
import torch
import torch.nn.functional as F
import numpy as np
from helpers import one_hot
from pretty_midi import PrettyMIDI, Instrument
import midi_input
import pdb

class GeneratorError(Exception):
    pass

def write_midi(note_sequence, output_dir, filename):

    #make output directory
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    #generate midi
    midi = PrettyMIDI()
    piano_track = Instrument(program=0, is_drum=False, name=filename)
    piano_track.notes = note_sequence
    midi.instruments.append(piano_track)
    output_name = output_dir + f"{filename}.midi"
    midi.write(output_name)

def sample(model, sample_length, prime_sequence=[], temperature=1,
        topk=None):
    """
    Generate a MIDI event sequence of a fixed length by randomly sampling from a model's distribution of sequences. Optionally, "seed" the sequence with a
    prime. A well-trained model will create music that responds to the prime
    and develops upon it.
    """
    #deactivate training mode
    model.eval()
    if len(prime_sequence) == 0:
        #if no prime is provided, randomly select a starting event
        input_sequence = [np.random.randint(model.n_states)]
    else:
        input_sequence = prime_sequence

    for i in range(sample_length):
        if torch.cuda.is_available():
            input_tensor = torch.LongTensor(input_sequence).cuda()
        else:
            input_tensor = torch.LongTensor(input_sequence)
        #add singleton dimension for the batch 
        input_tensor = input_tensor.unsqueeze(0)
        out = model(input_tensor)
        probs = F.softmax(out / temperature, dim=-1)
        #keep the probability distribution for the *next* state only
        probs = probs[:, -1, :]

        if topk is not None:
            #sample from only the top k most probable states
            values, indices = probs.topk(topk)
            if torch.cuda.is_available():
                zeros = torch.zeros(model.n_states).cuda()
            else:
                zeros = torch.zeros(model.n_states)
            probs = torch.scatter(zeros, 0, indices, values)

        next_char_ix = torch.multinomial(probs,1).item()

        input_sequence.append(next_char_ix)

    return input_sequence

def main():
    parser = argparse.ArgumentParser("Script to generate MIDI tracks by sampling from a trained model.")

#    parser.add_argument("--model_key", type=str, 
#            help="key to MODEL_DICT, allowing access to the path of a saved model & its params")
    parser.add_argument("--sample_length", type=int, default=512,
            help="number of events to generate")
    parser.add_argument("--temps", nargs="+", type=float, 
            default=[1.0],
            help="space-separated list of temperatures to use when sampling")
    parser.add_argument("--topks", nargs="+", type=int,
            help="space-separated list of topks to use when sampling")
    parser.add_argument("--n_trials", type=int, default=5,
            help="number of MIDI samples to generate per experiment")
    parser.add_argument("--live_input", action='store_true', default = False,
            help="if true, take in a seed from a MIDI input controller")

    parser.add_argument("--play_live", action='store_true', default=False,
            help="play sample(s) at end of script if true")
    parser.add_argument("--keep_ghosts", action='store_true', default=False)
    parser.add_argument("--stuck_note_duration", type=int, default=0)

    args=parser.parse_args()

#    model_key = args.model_key
#    if MODEL_DICT.get(model_key) is None:
#        raise GeneratorError("model key not supplied or not recognized!")
    model_path = "saved_models/tf_20200124"
    model_key = "tf_20200124"
    model_args = {"n_states": 413, "d_model": 64,
            "dim_feedforward": 512, "n_heads": 4, "n_layers": 3}
    try:
        state = torch.load(model_path)
    except RuntimeError:
        state = torch.load(model_path, map_location="cpu")
    
    n_velocity_events = 32
    n_time_shift_events = 125

    decoder = SequenceEncoder(n_time_shift_events, n_velocity_events)

    if args.live_input:
        print("Expecting a midi input...")
        note_sequence = midi_input.read(n_velocity_events, n_time_shift_events)
        prime_sequence = decoder.encode_sequences([note_sequence])[0]

    else:
        prime_sequence = []

    model = MusicTransformer(**model_args)
    model.load_state_dict(state)

    temps = args.temps

    topks = args.topks
    if topks is None:
        topks = [None]

    trial_key = str(uuid.uuid4())[:6]
    n_trials = args.n_trials

    keep_ghosts = args.keep_ghosts
    stuck_note_duration = None if args.stuck_note_duration == 0 else args.stuck_note_duration

    for temp in temps:
        print(f"sampling temp={temp}")
        note_sequence = []
        for i in range(n_trials):
            print("generating sequence")
            output_sequence = sample(model, prime_sequence = prime_sequence,
                    sample_length=args.sample_length, temperature=temp)
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
