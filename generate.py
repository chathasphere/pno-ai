import argparse, pathlib, uuid, subprocess
import model
from sequence_encoder import SequenceEncoder
import torch
import torch.nn.functional as F
import numpy as np
from helpers import one_hot
from pretty_midi import PrettyMIDI, Instrument
import midi_input

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
    #initialize output
    output_sequence = prime_sequence
    if len(prime_sequence) == 0:
        #if no prime is provided, randomly select a starting event
        input_sequence = [np.random.randint(model.n_states)]
    else:
        input_sequence = prime_sequence
    #initialize hidden state
    hx = None
    for i in range(sample_length):
        input_tensor = one_hot(input_sequence, model.n_states)
        #add singleton dimension for the batch (in the middle)
        input_tensor = input_tensor.unsqueeze(1)
        #verify batching works
        out, hx = model(input_tensor, hx, pack_batches=True)
        hx = tuple(h.detach() for h in hx)

        #out = out.squeeze()
        probs = F.softmax(out / temperature, dim=-1)
        #keep the probability distribution for the *next* state only
        probs = probs[-1,:]

        if topk is not None:
            #sample from only the top k most probable states
            values, indices = probs.topk(topk)
            if torch.cuda.is_available():
                zeros = torch.zeros(model.n_states).cuda()
            else:
                zeros = torch.zeros(model.n_states)
            probs = torch.scatter(zeros, 0, indices, values)

        next_char_ix = torch.multinomial(probs,1).item()

        input_sequence = [next_char_ix]
        output_sequence.append(next_char_ix)

    return output_sequence


#SAVE INFORMATION ON MODELS HERE
#should this be a yaml? perhaps.
MODEL_DICT = {
        'rnn': {'path': "saved_models/rnn_20190916", 
            'model': model.MusicRNN, 
            'args': {'n_states': 413, 'hidden_size': 512, 'n_rnn_layers': 3},
            'n_velocity_events': 32,
            'n_time_shift_events': 125}
        }


def main():
    parser = argparse.ArgumentParser("Script to generate MIDI tracks by sampling from a trained model.")

    parser.add_argument("--model_key", type=str, 
            help="key to MODEL_DICT, allowing access to the path of a saved model & its params")
    parser.add_argument("--sample_length", type=int, default=512,
            help="number of events to generate")
    parser.add_argument("--temps", nargs="+", type=float, default=1.0,
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

    model_key = args.model_key
    if MODEL_DICT.get(model_key) is None:
        raise GeneratorError("model key not supplied or not recognized!")

    try:
        state = torch.load(MODEL_DICT[model_key]['path'])
    except RuntimeError:
        state = torch.load(MODEL_DICT[model_key]['path'], map_location="cpu")
    
    n_velocity_events = MODEL_DICT[model_key]['n_velocity_events']
    n_time_shift_events = MODEL_DICT[model_key]['n_time_shift_events']

    decoder = SequenceEncoder(n_time_shift_events, n_velocity_events)

    if args.live_input:
        print("Expecting a midi input...")
        note_sequence = midi_input.read(n_velocity_events, n_time_shift_events)
        prime_sequence = decoder.encode_sequences([note_sequence])[0]

    else:
        prime_sequence = []

    model_args = MODEL_DICT[model_key]['args']
    model = MODEL_DICT[model_key]['model'](**model_args)
    model.load_state_dict(state)

    temps = args.temps

    topks = args.topks
    if topks is None:
        topks = [None]

    trial_key = str(uuid.uuid4())[:6]
    n_trials = args.n_trials

    keep_ghosts = args.keep_ghosts
    stuck_note_duration = None if args.stuck_note_duration == 0 else args.stuck_note_duration

    #TODO take in a priming sequence
    for temp in temps:
        print(f"sampling temp={temp}")
        note_sequence = []
        for i in range(n_trials):
            output_sequence = sample(model, prime_sequence = prime_sequence,
                    sample_length=args.sample_length, temperature=temp)
            note_sequence = decoder.decode_sequence(output_sequence, 
                verbose=True, stuck_note_duration=3)


        #note_sequence2 = decoder.decode_sequence(output_sequence, stuck_note_duration=10)

        #note_sequence3 = decoder.decode_sequence(output_sequence, keep_ghosts=True)
            output_dir = f"output/{model_key}/{trial_key}/"
            file_name = f"sample{i+1}_{temp}"
            write_midi(note_sequence, output_dir, file_name)
        #write_midi(note_sequence, output_dir, "sticky_notes")
        #write_midi(note_sequence, output_dir, "ghost_notes")

    for temp in temps:      
        try:
            subprocess.run(['timidity', f"output/{model_key}/{trial_key}/sample{i+1}_{temp}.midi"])
        except KeyboardInterrupt:
            continue

if __name__ == "__main__":
    main()
