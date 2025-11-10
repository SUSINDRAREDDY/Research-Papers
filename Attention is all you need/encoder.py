import torch
import torch.nn as nn
import math
from encoder_layer import EncoderLayer
from positional_encoding import PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, hidden_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.tok_embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, hidden_layers, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        # Embedding with scaling (important for stability)
        x = self.tok_embedding(src) * math.sqrt(self.d_model)
        # Add positional encoding
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)

        return x
