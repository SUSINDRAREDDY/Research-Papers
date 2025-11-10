import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import FNN

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_layers, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout) #Masked Self-Attention (decoder looks only at previous tokens)
        self.cross_atten = MultiHeadAttention(d_model, num_heads, dropout) #Cross-Attention (decoder attends to encoder output)
        self.ffn = FNN(d_model, hidden_layers, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        _x = x
        x = self.self_attn(x,x,x,tgt_mask) # Decoder attends only to past words
        x = _x + self.dropout1(x)
        x = self.norm1(x)

        _x = x
        x = self.cross_atten(x, enc_output, enc_output, src_mask) #We take K,V from encoder and Q from decoder Masked attn
        x = _x + self.dropout2(x)
        x = self.norm2(x)

        _x = x
        x = self.ffn(x)
        x = _x + self.dropout3(x)
        x = self.norm3(x)

        return x
