import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(q,k,v, mask=None, dropout=None):
    # q, k, v: (batch_size, num_heads, q_seq_len, d_k) and (batch_size, num_heads, k_seq_len, d_k)
    # d_k = d_model / num_heads = 512 / 8 = 64 from paper

    d_k = q.size(-1)
    q_seq_len = q.size(2)
    k_seq_len = k.size(2)
    scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k) # (batch_size, num_heads, q_seq_len, k_seq_len)
    
    if mask is not None:
        if mask.dim() == 3:
            if mask.size(1) == 1:
                # Padding mask: (batch_size, 1, mask_seq_len)
                mask_seq_len = mask.size(2)
                if q_seq_len == k_seq_len and mask_seq_len == q_seq_len:
                    # Self-attention: mask both query and key positions
                    mask = mask.unsqueeze(2) & mask.unsqueeze(3)  # (batch_size, 1, q_seq_len, k_seq_len)
                else:
                    # Cross-attention or mask for key positions only: expand to mask key positions (columns)
                    mask = mask.unsqueeze(2)  # (batch_size, 1, 1, k_seq_len)
            else:
                # Causal mask: (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax with numerical stability
    attn = torch.softmax(scores, dim=-1) #attn weights
    
    attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
    
    if dropout is not None:
        attn = dropout(attn)
    
    output = torch.matmul(attn, v)
    return output, attn

#Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0 #check if num_heads can evenly divde d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q,k,v, mask=None):
        batch_size = q.size(0)

        #now we split the data into n heads
        q=self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)  # (batch_size, heads, seq_len, d_k)
        k=self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)  # (batch_size, heads, seq_len, d_k)
        v=self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)  # (batch_size, heads, seq_len, d_k)

        output, attn = scaled_dot_product_attention(q,k,v,mask, self.dropout)

        #here concat all heads
        concat = output.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads*self.d_k)  #(batch_size, seq_len, num_heads * d_k) = (batch_size, seq_len, d_model)
        output = self.out(concat)  # (batch_size, seq_len, d_model)
        return output