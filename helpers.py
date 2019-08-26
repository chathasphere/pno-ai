import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pdb

def one_hot(sequence, n_states):
    """
    Given a list of integers and the maximal number of unique values found
    in the list, return a one-hot encoded tensor of shape (m, n)
    where m is sequence length and n is n_states.
    """
    return torch.eye(n_states)[sequence,:]


def prepare_batches(sequences, batch_size, n_states, sequence_length):
    """
    Converts a list of numeric sequences to one-hot encoded batches of the following shape:
    (sequence_length x batch_size x n_states),
    where sequence_length is the length of the longest sequence (this function pads/packs shorter sequences)
    and n_states is the dimensionality of the one-hot encoding.

    Returns a list of Pytorch tensors.
    """
    sequences = sorted(sequences, key = lambda x: len(x), reverse=True)
    batches = []
    n_sequences = len(sequences)
    for i in range(0, n_sequences, batch_size):
        batches.append(sequences[i:i+batch_size])

    packed_batches = []
    for batch in batches:
        sequence_lengths = [len(s) for s in batch]
        coded_batch = [one_hot(s, n_states) for s in batch]

        padded_batch = pad_sequence(coded_batch)
        packed_batches.append(pack_padded_sequence(padded_batch, sequence_lengths))

    return packed_batches




