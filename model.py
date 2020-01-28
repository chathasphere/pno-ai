import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import clones, d
from attention import MultiheadedAttention
import math

class MusicTransformer(nn.Module):
    """Generative, autoregressive transformer model. Train on a 
    dataset of encoded musical sequences."""

    def __init__(self, n_tokens, d_model=64, n_heads=4, depth=2, d_feedforward=512, dropout=0.1):
        """
        Args:
            n_tokens: number of commands/states in encoded musical sequence
            d_model: dimensionality of embedded sequences
            n_heads: number of attention heads
            depth: number of stacked transformer layers
            d_feedforward: dimensionality of dense sublayer 
            dropout: probability of dropout in dropout sublayer
        """
        super().__init__()
        #number of commands in an encoded musical sequence
        self.n_tokens = n_tokens
        #embedding layer
        self.embed = SequenceEmbedding(n_tokens, d_model)
        #positional encoding layer
        self.pos = PositionalEncoding(d_model, dropout)
        #last layer, outputs logits of next token in sequence
        self.to_scores = nn.Linear(d_model, n_tokens)
        self.layers = clones(DecoderLayer(d_model, n_heads,
            d_feedforward, dropout), depth)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.pos(x)
        #pass input batch and mask through layers
        for layer in self.layers:
            x  = layer(x, mask)
        #one last normalization for good measure
        z = self.norm(x)
        return self.to_scores(z)

class DecoderLayer(nn.Module):

    def __init__(self, size, n_heads, d_feedforward, dropout):

        super().__init__()
        self.self_attn = MultiheadedAttention(n_heads, size)
        self.feed_forward = PositionwiseFeedForward(size, d_feedforward, dropout)
        self.size = size
        #normalize over mean/std of embedding dimension
        self.norm1 = nn.LayerNorm(size)
        self.norm2 = nn.LayerNorm(size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x, mask):
        #perform masked attention on input
        #masked so queries cannot attend to subsequent keys
        #Pass through sublayers of attention and feedforward.
        #Apply dropout to sublayer output, add it to input, and norm.
        attn = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn)
        x = self.norm1(x)

        ff = self.feed_forward(x)
        x = x + self.dropout2(ff)
        x = self.norm2(x)

        return x

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
    Fixed sinusoidal positional encodings. This contains information
    on the relative position of tokens. Each position is a sinusoidal wave, and each offset can be expressed as a linear function of another position.  
    """
    def __init__(self, d_model, dropout, max_len=5000):

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
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
