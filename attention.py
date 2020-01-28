import torch
import torch.nn as nn
import math
from helpers import clones
import torch.nn.functional as F

class AttentionError(Exception):
    pass

class MultiheadedAttention(nn.Module):
    
    def __init__(self, n_heads, d_model, dropout=0.1):
        
        super().__init__()
        self.n_heads = n_heads
        if d_model % n_heads != 0:
            raise AttentionError("Number of heads does not divide model dimension")
        self.d_k = d_model // n_heads
        #linear layers receiving queries, keys, values
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.final_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, query, key, value, mask=None):
        #batch size, length, embedding dimension
        b,t,e = query.size()
        #split batches into multiple heads
        query, key, value = [linear(x).view(b, -1, self.n_heads, self.d_k).transpose(1,2)
                for linear, x in zip(self.linears, (query, key, value))]
        #(batch_size, n_heads, sequence_length, d_k)
        #(b, h, t, e/h)
        if mask is not None:
            #same mask for all heads
            mask = mask.unsqueeze(1)
        
        #Scaled Dot Product Attention over all heads
        #Get dot products of queries with keys to measure compatibility
        scores = torch.matmul(query, key.transpose(-2,-1))
        #scores: (b, h, t, t)
        #Interpreted as logits
        if mask is not None:
            #mask out illegal connections with neg infty
            scores = scores.masked_fill(mask == 0, -1e9)
        #Softmax to convert to probabilities
        p_attn = F.softmax(scores, dim = -1)
        #concatenate heads
        p_attn = self.dropout(p_attn)
        #output is an average of values, weighted by attention
        x = torch.matmul(p_attn, value)
        x = x.transpose(1,2).contiguous() \
                .view(b, -1, self.n_heads * self.d_k)

        return self.final_linear(x)



