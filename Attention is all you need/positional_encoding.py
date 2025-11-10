import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model) #(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #(max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) #only for even indices
        pe[:, 1::2] = torch.cos(position * div_term) #only for odd indices
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model) ~ (batch_size, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding and apply dropout
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)