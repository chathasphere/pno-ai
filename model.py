import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import PackedSequence
from helpers import clones
from attention import MultiheadedAttention
import math
import pdb

class MusicTransformer(nn.Module):

    def __init__(self, n_states, d_model=64, n_heads=4, n_layers=2, dim_feedforward=512, dropout=0.1):

        super().__init__()

        self.decoder = Decoder(DecoderLayer(d_model, n_heads, dim_feedforward, dropout), n_layers)

        self.embed = nn.Sequential(SequenceEmbedding(n_states, d_model), PositionalEncoding(d_model, dropout))

        #number of unique states in a musical sequence
        self.n_states = n_states
        #project embedded sequence back onto space of sequence states
        #interpret output as logits
        self.to_scores = nn.Linear(d_model, n_states)
    
    def forward(self, x, mask=None):

        z = self.decoder(self.embed(x), mask)

        return self.to_scores(z)

class Decoder(nn.Module):
    "Stack of n decoder layers"
    def __init__(self, layer, n_layers):

        super().__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)


    def forward(self, x, mask):
        "Pass input and mask through each layer"
        for layer in self.layers:
            x = layer(x, mask)
        #normalize output of decoder
        return self.norm(x)

class DecoderLayer(nn.Module):

    def __init__(self, size, n_heads, dim_feedforward, dropout):

        super().__init__()
        self.self_attn = MultiheadedAttention(n_heads, size)
        self.feed_forward = PositionwiseFeedForward(size, dim_feedforward, dropout)
        self.size = size
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x, mask):
        #perform masked attention on input
        #masked so queries cannot attend to subsequent keys
        attn = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn)
        x = self.norm1(x)

        ff = self.feed_forward(x)
        x = x + self.dropout2(ff)
        x = self.norm2(x)

        return x


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
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
