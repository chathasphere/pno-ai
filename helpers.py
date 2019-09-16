import torch
import numpy as np
from pretty_midi import Note
import torch.nn.functional as F

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

    return output_seqeunce

