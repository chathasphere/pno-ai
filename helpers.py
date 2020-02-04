import torch
import numpy as np
from pretty_midi import Note, PrettyMIDI, Instrument
import torch.nn.functional as F
import copy, pathlib

def vectorize(sequence):
    """
    Converts a list of pretty_midi Note objects into a numpy array of
    dimension (n_notes x 4)
    """
    array = [[note.start, note.end, note.pitch, note.velocity] for
            note in sequence]
    return np.asarray(array)

def devectorize(note_array):
    """
    Converts a vectorized note sequence into a list of pretty_midi Note
    objects
    """
    return [Note(start = a[0], end = a[1], pitch=a[2],
        velocity=a[3]) for a in note_array.tolist()]


def one_hot(sequence, n_states):
    """
    Given a list of integers and the maximal number of unique values found
    in the list, return a one-hot encoded tensor of shape (m, n)
    where m is sequence length and n is n_states.
    """
    if torch.cuda.is_available():
        return torch.eye(n_states)[sequence,:].cuda()
    else:
        return torch.eye(n_states)[sequence,:]

def decode_one_hot(vector):
    '''
    Given a one-hot encoded vector, return the non-zero index
    '''
    return vector.nonzero().item()

def prepare_batches(sequences, batch_size):
    """
    Splits a list of sequences into batches of a fixed size. Each sequence yields an input sequence
    and a target sequence, with the latter one time step ahead. For example, the sequence "to be or not
    to be" gives an input sequence of "to be or not to b" and a target sequence of "o be or not to be."
    """
    n_sequences = len(sequences)
    for i in range(0, n_sequences, batch_size):
        batch = sequences[i:i+batch_size]
	#needs to be in sorted order for packing batches to work
        batch = sorted(batch, key = len, reverse=True)
        input_sequences, target_sequences = [], []

        for sequence in batch:
            input_sequences.append(sequence[:-1])
            target_sequences.append(sequence[1:])

        yield input_sequences, target_sequences

def clones(module, N):
    "Clone N identical layers of a module"
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def d(tensor=None):
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

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

def sample(model, sample_length, prime_sequence=[], temperature=1):
    """
    Generate a MIDI event sequence of a fixed length by randomly sampling from a model's distribution of sequences. Optionally, "seed" the sequence with a prime. A well-trained model will create music that responds to the prime and develops upon it.
    """
    #deactivate training mode
    model.eval()
    if len(prime_sequence) == 0:
        #if no prime is provided, randomly select a starting event
        input_sequence = [np.random.randint(model.n_tokens)]
    else:
        input_sequence = prime_sequence.copy()

    #add singleton dimension for the batch
    input_tensor = torch.LongTensor(input_sequence).unsqueeze(0)

    for i in range(sample_length):
        #select probabilities of *next* token
        out = model(input_tensor)[0, -1, :]
        #out is a 1d tensor of shape (n_tokens)
        probs = F.softmax(out / temperature, dim=0)
        #sample prob distribution for next character
        c = torch.multinomial(probs,1)
        input_tensor = torch.cat([input_tensor[:,1:], c[None]], dim=1)
        input_sequence.append(c.item())

    return input_sequence

