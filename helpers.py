import torch
import numpy as np
from pretty_midi import Note

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

#def sequences_to_tensor(sequences, pack_sequences, is_input, one_hot=:
#    #TODO figure out how CUDA is to work with target sequences
#    if is_input:
#        tensors = [one_hot(sequence, n_states) for sequence in sequences]
#    else:
#        tensors = [torch.tensor(s) for s in sequences]
#
#    padded_sequences = pad_sequence(tensors, batch_first = not(pack_sequences))
#
#    if pack_sequences:
#        sequence_lengths = [len(s) for s in sequences]
#        return packed_padded_sequence(padded_sequences, sequence_lengths)
#    else:
#        return padded_sequences
#
