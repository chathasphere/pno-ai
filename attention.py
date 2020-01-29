import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from helpers import d

class AttentionError(Exception):
    pass

class MultiheadedAttention(nn.Module):
    """
    Narrow multiheaded attention. Each attention head inspects a 
    fraction of the embedding space and expresses attention vectors for each sequence position as a weighted average of all (earlier) positions.
    """

    def __init__(self, d_model, heads=8, dropout=0.1):

        super().__init__()
        if d_model % heads != 0:
            raise AttentionError("Number of heads does not divide model dimension")
        self.d_model = d_model
        self.heads = heads
        s = d_model // heads
        self.linears = torch.nn.ModuleList([nn.Linear(s, s, bias=False) for i in range(3)])
        self.recombine_heads = nn.Linear(heads * s, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        #batch size, sequence length, embedding dimension
        b, t, e = x.size()
        h = self.heads
        #each head inspects a fraction of the embedded space
        #head dimension
        s = e // h
        x = x.view(b,t,h,s)
        #Apply weights to inputs and combine heads with batches
        queries, keys, values = [linear(x).transpose(1,2).\
                                 contiguous().view(b*h, t, s)
          for linear, x in zip(self.linears, (x,x,x))]
        #Compute scaled dot-product self-attention
        #scale pre-matrix multiplication   
        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))

        # mask out the upper half of the dot matrix, excluding the diagonal
        scores = torch.bmm(queries, keys.transpose(1, 2))
        subsequent_mask = torch.triu(torch.ones(1, t, t, device=d()), 0)
        scores = scores.masked_fill(subsequent_mask.transpose(1,2) == 0, -1e9)
        
        #Convert scores to probabilities
        attn_probs = F.softmax(scores, dim=2)
        attn_probs = self.dropout(attn_probs)
        #use attention to get a weighted average of values
        out = torch.bmm(attn_probs, values).view(b, h, t, s)
        #transpose and recombine attention heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)
        #last linear layer of weights
        return self.recombine_heads(out)
