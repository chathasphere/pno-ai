import torch
import torch.nn as nn
import pdb
import math
from helpers import clones
import torch.nn.functional as F

class AttentionError(Exception):
    pass

def attention(query, key, value, mask=None, dropout=None):
    """Scaled Dot Product Attention. Assumes matrices have batch dimension first"""
    d_k = query.size(-1)
    #scores are "compatibility" of queries with keys
    #matmul always multiplies across last two dimensions
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
    #scores has dim (batch_size, n_heads, seq_length, seq_length)
    if mask is not None:
        #mask out "illegal" connections with large negative values
        scores = scores.fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    #weighted sum of values, weights are normalized scores
    # 
    return torch.matmul(p_attn, value), p_attn


class MultiheadedAttention(nn.Module):
    
    def __init__(self, n_heads, d_model, dropout=0.1):
        
        super().__init__()
        self.n_heads = n_heads
        if d_model % n_heads != 0:
            raise AttentionError("Number of heads does not divide model dimension")
        self.d_k = d_model // n_heads
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.final_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        query, key, value = [linear(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
                for linear, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask = mask,
            dropout = self.dropout)

        x = x.transpose(1,2).contiguous() \
                .view(batch_size, -1, self.n_heads * self.d_k)

        return self.final_linear(x)



