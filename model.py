import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import PackedSequence
import pdb

class MusicRNN(nn.Module):

    def __init__(self, n_states, hidden_size, n_rnn_layers=1, dropout=0, batch_first=False):

        super().__init__()
        
        self.n_states = n_states
        self.lstm = nn.LSTM(n_states, hidden_size, n_rnn_layers, dropout=dropout,
                batch_first=batch_first)
        #projects RNN output back down onto the space of sequence states
        self.dense = nn.Linear(hidden_size, n_states)

    def forward(self, x, hx, pack_batches=True):


        recurrent_output, hx = self.lstm(x, hx)

        if pack_batches:
            #just extract the flattened (2d) packed tensor,
            #batch sizes are unneccessary.
            recurrent_output = recurrent_output[0]
        else:
            recurrent_output = recurrent_output.flatten(0,1)

        linear_output = self.dense(recurrent_output)

        return linear_output, hx


