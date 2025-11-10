import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import FNN

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_layers, dropout=0.1):
        super().__init__()
        self.self_attn=MultiHeadAttention(d_model, num_heads, dropout) # Multi-Head Self-Attention
        self.fnn = FNN(d_model,hidden_layers,dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        #x -> (batch_size, seq_len, d_model)
        attn_out = self.self_attn(x,x,x,mask)
        x = x+ self.dropout1(attn_out) # Residual connection
        x = self.norm1(x) # LayerNorm after residual

        fnn = self.fnn(x)
        x = x + self.dropout2(fnn) # Residual connection
        x = self.norm2(x) # LayerNorm after residual 

        return x