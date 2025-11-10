import torch
import torch.nn as nn
import math
from decoder_layer import DecoderLayer
from positional_encoding import PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, num_layers, num_heads, hidden_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.tok_embedding = nn.Embedding(output_dim, d_model) # Input: (batch_size, tgt_seq_len) -> Embedding -> (batch_size, tgt_seq_len, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, hidden_layers, dropout)for _ in range(num_layers)]) #stacking decoders

        self.norm = nn.LayerNorm(d_model)
        self.generator = nn.Linear(d_model, output_dim) #(batch_size, tgt_seq_len, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        # tgt: (batch_size, tgt_seq_len) -- token ids for target input
        # enc_output: (batch_size, src_seq_len, d_model) -- output from encoder
        # src_mask: (batch_size, 1, src_seq_len) or (batch_size, src_seq_len, src_seq_len)
        # tgt_mask: (batch_size, tgt_seq_len, tgt_seq_len) or causal mask

        # Embedding with scaling (important for stability)
        x = self.tok_embedding(tgt) * math.sqrt(self.d_model)
        # Add positional encoding
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        x = self.norm(x)
        logits = self.generator(x)
        return logits
