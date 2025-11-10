# transformer.py

import torch
import torch.nn as nn
import math
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_layers=4, num_heads=4, hidden_layers=512, dropout=0.1):
        super().__init__()
        # Encoder and Decoder stacks
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, hidden_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, hidden_layers, dropout)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: (batch_size, src_seq_len) -- input token ids
        tgt: (batch_size, tgt_seq_len) -- target token ids (shifted with <sos>)
        src_mask: mask for src (batch_size, src_seq_len) or (batch_size, 1, src_seq_len)
        tgt_mask: mask for tgt (batch_size, tgt_seq_len, tgt_seq_len)

        returns: logits (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # 1) Encode
        enc_output = self.encoder(src, src_mask)  # (batch_size, src_seq_len, d_model)

        # 2) Decode
        logits = self.decoder(tgt, enc_output, src_mask, tgt_mask)  # (batch_size, tgt_seq_len, tgt_vocab_size)
        return logits

def make_src_mask(src_batch, pad_idx):
    # mask shape: (batch_size, src_seq_len)
    mask = (src_batch != pad_idx).unsqueeze(1)
    return mask


def make_tgt_mask(tgt_batch, pad_idx):
    batch_size, tgt_len = tgt_batch.size()
    pad_mask = (tgt_batch != pad_idx).unsqueeze(1)  # (batch_size,1,tgt_len)
    subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=tgt_batch.device))
    mask = pad_mask & subsequent_mask.unsqueeze(0)  # (batch_size, tgt_len, tgt_len)
    return mask