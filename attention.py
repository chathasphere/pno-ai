import torch
import torch.nn as nn
import pdb

class AttentionError(Exception):
    pass

class MultiheadedAttention(nn.Module):
    
    def __init__(self, n_heads, d_model, dropout=0.1):
        
        super().__init__()
        self.n_heads = n_heads
        if d_model % n_heads != 0:
            raise AttentionError("Number of heads does not divide model dimension")
        self.d_k = d_model / n_heads

    def forward(self, query, key, value, mask=None):
        pdb.set_trace()

