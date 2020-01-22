import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import PackedSequence
from helpers import clones
from attention import MultiheadedAttention
import math

class MusicRNN(nn.Module):
    """
    Recurrent neural network using Long Short Term Memory (LSTM) layer(s) to
    model temporal relations over encoded event sequences (see sequence_encoder.py for details.) 
    """

    def __init__(self, n_states, hidden_size, n_rnn_layers=1, dropout=0, batch_first=False):
        """
        args:
            n_states (int): Number of unique events in the MIDI encoding. This is the number of features the model is attempting to predict, sequentially.
           hidden_size (int): Number of neurons in the LSTM, this defines the dimension of the recurrent hidden state.
           n_rnn_layers (int): Number of stacked RNN layers
           dropout (float): Proportion (0 to 1) of neural connections to drop during the LSTM's optional dropout phase
           batch_first (bool): If true, model input is of shape 
           (batch_size x n_states x sequence_length). Else (this is required for packed, mixed-length batches) 
           (n_states x batch_size x sequence length).
        """

        super().__init__()
        
        self.n_states = n_states
        self.lstm = nn.LSTM(n_states, hidden_size, n_rnn_layers, dropout=dropout,
                batch_first=batch_first)
        #projects RNN output back down onto the space of sequence states
        self.dense = nn.Linear(hidden_size, n_states)

    def forward(self, x, hx, pack_batches=True):
        """
        Forward call of the RNN. Input is passed through the recurrent LSTM layer(s) and then "decoded" by a dense, linear layer which projects it back down onto the space of possible events.

        Returns:
            linear_output (tensor): Unnormalized log probabilities predicting next character for every sequence in the batch, *concatenated* (this makes the loss calculation possible)
            hx (tensor): Hidden state of RNN. This is the model's "memory" and
            gets reused in subsequent minibatches.
        """
        recurrent_output, hx = self.lstm(x, hx)


        if pack_batches:
            #just extract the flattened (2d) packed tensor,
            #batch sizes are unneccessary.
            recurrent_output = recurrent_output[0]
        else:
            recurrent_output = recurrent_output.flatten(0,1)


        linear_output = self.dense(recurrent_output)

        return linear_output, hx

class MusicTransformer(nn.Module):

    def __init__(self, n_states, d_model=256, n_heads=8, n_encoder_layers=6, n_decoder_layers=6,
            dim_feedforward=2048, dropout=0.1):

        super().__init__()

        self.encoder = Encoder(EncoderLayer(d_model, n_heads, dim_feedforward,
            dropout), n_encoder_layers)
        
        self.decoder = Decoder(DecoderLayer(d_model, n_heads, dim_feedforward,
            dropout), n_decoder_layers)

        self.src_embed = nn.Sequential(SequenceEmbedding(n_states, d_model), PositionalEncoding(d_model, dropout))

        self.tgt_embed = nn.Sequential(SequenceEmbedding(n_states, d_model), PositionalEncoding(d_model, dropout))
        
        #number of unique states in a musical sequence
        self.n_states = n_states
        #project embedded sequence back onto space of sequence states
        self.dense = nn.Linear(d_model, n_states)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encoder(self.src_embed(src), src_mask)

        output = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

        projected_output = self.dense(output)

        return projected_output

class Encoder(nn.Module):
    "Stack of n encoder layers"
    def __init__(self, layer, n_layers):

        super().__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)


    def forward(self, x, mask):
        "Pass input and mask through each layer"
        for layer in self.layers:
            x = layer(x, mask)
        #normalize output of encoder
        return self.norm(x)

class EncoderLayer(nn.Module):

    def __init__(self, size, n_heads, dim_feedforward, dropout):

        super().__init__()
        self.self_attn = MultiheadedAttention(n_heads, size)
        self.feed_forward = PositionwiseFeedForward(size, dim_feedforward, dropout)
        self.size = size
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, src, src_mask):
        attn = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(attn)
        src = self.norm1(src)

        ff = self.feed_forward(src)
        src = src + self.dropout2(ff)
        src = self.norm2(src)

        return src

class Decoder(nn.Module):
    def __init__(self, layer, n_layers):

        super().__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):

    def __init__(self, size, n_heads, dim_feedforward, dropout):

        super().__init__()
        self.self_attn = MultiheadedAttention(n_heads, size)
        self.src_attn = MultiheadedAttention(n_heads, size)
        self.feed_forward = PositionwiseFeedForward(size, dim_feedforward, dropout)
        self.size = size
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, src_mask, tgt_mask):
        #prevent positions from attending to future positions with tgt_mask
        self_attn = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(self_attn)
        tgt = self.norm1(tgt)
        #perform attention over encoder output
        #query target embeddings, key/values are encoder output
        src_attn = self.src_attn(tgt, memory, memory, src_mask)
        tgt = tgt + self.dropout2(src_attn)
        tgt = self.norm2(tgt)

        ff = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(ff)
        tgt = self.norm3(tgt)

        return tgt

class LayerNorm(nn.Module):
    def __init__(self, n_features, eps=1e-6):

        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_features))
        self.beta = nn.Parameter(torch.zeros(n_features))
        self.eps = eps

    def forward(self, x):
        #implies batch is the last dimension?
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class SequenceEmbedding(nn.Module):
    """
    Standard embedding, scaled by the sqrt of model's hidden state size
    """
    def __init__(self, vocab_size, model_size):
        super().__init__()
        self.d_model = model_size
        self.emb = nn.Embedding(vocab_size, model_size)

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encodings
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        #
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        #geometric progression of wavelengths
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * \
                - (math.log(10000.0) / d_model))
        #even positions
        pe[0:, 0::2] = torch.sin(position * div_term)
        #odd positions
        pe[0:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        #save untrained parameter to state_dict
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.Tensor(self.pe[:, :x.size(1)])
        return self.dropout(x)
