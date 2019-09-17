import argparse, pdb, pathlib, uuid
from model import MusicRNN
from sequence_encoder import SequenceEncoder
import torch
import torch.nn.functional as F
import numpy as np
from helpers import one_hot
from pretty_midi import PrettyMIDI, Instrument


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
        #add singleton dimension for the batch
        input_tensor = input_tensor.unsqueeze(0)
        #verify batching works
        out, hx = model(input_tensor, hx, pack_batches=False)

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




def main():
    parser = argparse.ArgumentParser("Script to generate MIDI tracks by sampling from a trained model.")

    parser.add_argument("--model_path", type=str, 
            help="path to saved state dict of a trained Pytorch model")
    parser.add_argument("--sample_length", type=int, default=512,
            help="number of events to generate")

    args=parser.parse_args()

    try:
        state = torch.load(args.model_path)
    except RuntimeError:
        state = torch.load(args.model_path, map_location="cpu")
    
    #TODO load a configuration in rather than hardcode this
    n_velocity_events = 32
    n_time_shift_events = 125
    n_states = 256 + n_velocity_events + n_time_shift_events

    decoder = SequenceEncoder(n_time_shift_events, n_velocity_events)

    model = MusicRNN(n_states, 512, n_rnn_layers=3)
    model.load_state_dict(state)

    #temps = [1, .75, .5]
    #temps = [1, .9, .8, .7, .6, .5] 
    temps = [1]

    #topks = [None, 100, 50]
    topks = [None]

    trial_key = str(uuid.uuid4())[:6]
    n_trials = 3
    #TODO take in a priming sequence
    for temp in temps:
        for topk in topks:
            print(f"sampling temp={temp}, topk={topk}...")
            for i in range(n_trials):
                output_sequence = sample(model, prime_sequence = [],
                        sample_length=args.sample_length, temperature=temp, 
                        topk=topk)
                note_sequence = decoder.decode_sequence(output_sequence, 
                    verbose=True, stuck_note_duration=1)

            #note_sequence2 = decoder.decode_sequence(output_sequence, stuck_note_duration=10)

            #note_sequence3 = decoder.decode_sequence(output_sequence, keep_ghosts=True)

                output_dir = f"output/rnn/{trial_key}/"
                file_name = f"sample{i+1}_{temp}_{topk}"
                write_midi(note_sequence, output_dir, file_name)
            #write_midi(note_sequence, output_dir, "sticky_notes")
            #write_midi(note_sequence, output_dir, "ghost_notes")

if __name__ == "__main__":
    main()
